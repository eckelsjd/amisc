"""Module for compression methods.

Especially useful for field quantities with high dimensions.

Includes:

- `Compression` — an interface for specifying a compression method for field quantities.
- `SVD` — a Singular Value Decomposition (SVD) compression method.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import RBFInterpolator

from amisc.serialize import PickleSerializable
from amisc.utils import relative_error

__all__ = ["Compression", "SVD"]


@dataclass
class Compression(PickleSerializable, ABC):
    """Base class for compression methods. Compression methods should:

    - `compute_map` - compute the compression map from provided data
    - `compress` - compress data into a latent space
    - `reconstruct` - reconstruct the compressed data back into the full space
    - `latent_size` - return the size of the latent space
    - `estimate_latent_ranges` - estimate the range of the latent space coefficients

    !!! Note "Specifying fields"
        The `fields` attribute is a list of strings that specify the field quantities to compress. For example, for
        3D velocity data, the fields might be `['ux', 'uy', 'uz']`. The length of the
        `fields` attribute is used to determine the number of quantities of interest at each grid point in `coords`.
        Note that interpolation to/from the compression grid will assume a shape of `(num_pts, num_qoi)` for the
        states on the grid, where `num_qoi` is the length of `fields` and `num_pts` is the length of `coords`. When
        constructing the compression map, this important fact should be considered when passing data to
        `compute_map`.

    In order to use a `Compression` object, you must first call `compute_map` to compute the compression map, which
    should set the private value `self._map_computed=True`. The `coords` of the compression grid must also be
    specified. The `coords` should have the shape `(num_pts, dim)` where `num_pts` is the number of points in the
    compression grid and `dim` is the number of spatial dimensions. If `coords` is a 1d array, then the `dim` is
    assumed to be 1.

    :ivar fields: list of field quantities to compress
    :ivar method: the compression method to use (only svd is supported for now)
    :ivar coords: the coordinates of the compression grid
    :ivar interpolate_method: the interpolation method to use to interpolate to/from the compression grid
                              (only `rbf` (i.e. radial basis function) is supported for now)
    :ivar interpolate_opts: additional options to pass to the interpolation method
    :ivar _map_computed: whether the compression map has been computed
    """
    fields: list[str] = field(default_factory=list)
    method: str = 'svd'
    coords: np.ndarray = None  # (num_pts, dim)
    interpolate_method: str = 'rbf'
    interpolate_opts: dict = field(default_factory=dict)
    _map_computed: bool = False

    @property
    def map_exists(self):
        """All compression methods should have `coords` when their map has been constructed."""
        return self.coords is not None and self._map_computed

    @property
    def dim(self):
        """Number of physical grid coordinates for the field quantity, (i.e. x,y,z spatial dims)"""
        return self.coords.shape[1] if (self.coords is not None and len(self.coords.shape) > 1) else 1

    @property
    def num_pts(self):
        """Number of physical points in the compression grid."""
        return self.coords.shape[0] if self.coords is not None else None

    @property
    def num_qoi(self):
        """Number of quantities of interest at each grid point, (i.e. `ux, uy, uz` for 3d velocity data)."""
        return len(self.fields) if self.fields is not None else 1

    @property
    def dof(self):
        """Total degrees of freedom in the compression grid (i.e. `num_pts * num_qoi`)."""
        return self.num_pts * self.num_qoi if self.num_pts is not None else None

    def _correct_coords(self, coords):
        """Correct the coordinates to be in the correct shape for compression."""
        coords = np.atleast_1d(coords)
        if np.issubdtype(coords.dtype, np.object_):  # must be object array of np.arrays (for unique coords)
            for i, arr in np.ndenumerate(coords):
                if arr is None:
                    continue  # skip empty values
                if len(arr.shape) == 1:
                    coords[i] = arr[..., np.newaxis] if self.dim == 1 else arr[np.newaxis, ...]
        else:
            if len(coords.shape) == 1:
                coords = coords[..., np.newaxis] if self.dim == 1 else coords[np.newaxis, ...]
        return coords

    def interpolator(self):
        """The interpolator to use during compression and reconstruction. Interpolator expects to be used as:

        ```python
        xg = np.ndarray    # (num_pts, dim)  grid coordinates
        yg = np.ndarray    # (num_pts, ...)  scalar values on grid
        xp = np.ndarray    # (Q, dim)        evaluation points

        interp = interpolate_method(xg, yg, **interpolate_opts)

        yp = interp(xp)    # (Q, ...)        interpolated values
        ```
        """
        method = self.interpolate_method or 'rbf'
        match method.lower():
            case 'rbf':
                return RBFInterpolator
            case other:
                raise NotImplementedError(f"Interpolation method '{other}' is not implemented.")

    def interpolate_from_grid(self, states: np.ndarray, new_coords: np.ndarray):
        """Interpolate the states on the compression grid to new coordinates.

        :param states: `(*loop_shape, dof)` - the states on the compression grid
        :param new_coords: `(*coord_shape, dim)` - the new coordinates to interpolate to; if a 1d object array, then
                            each element is assumed to be a unique `(*coord_shape, dim)` array with assumed
                            same length as loop_shape of the states -- will interpolate each state to the
                            corresponding new coordinates
        :return: `dict` of `(*loop_shape, *coord_shape)` for each qoi - the interpolated states; will return a
                 single 1d object array for each qoi if new_coords is a 1d object array
        """
        new_coords = self._correct_coords(new_coords)
        grid_coords = self._correct_coords(self.coords)

        coords_obj_array = np.issubdtype(new_coords.dtype, np.object_)

        # Iterate over one set of coords and states at a time
        def _iterate_coords_and_states():
            if coords_obj_array:
                for index, c in np.ndenumerate(new_coords):  # assumes same number of coords and states
                    yield index, c, states[index]
            else:
                yield (0,), new_coords, states  # assumes same coords for all states

        all_qois = np.empty(new_coords.shape if coords_obj_array else (1,), dtype=object)

        # Do interpolation for each set of unique coordinates (if multiple)
        for j, n_coords, state in _iterate_coords_and_states():
            if n_coords is None:  # Skip empty coords
                continue

            skip_interp = (n_coords.shape == grid_coords.shape and np.allclose(n_coords, grid_coords))

            ret_dict = {}
            loop_shape = state.shape[:-1]
            coords_shape = n_coords.shape[:-1]
            state = state.reshape((*loop_shape, self.num_pts, self.num_qoi))
            n_coords = n_coords.reshape((-1, self.dim))
            for i, qoi in enumerate(self.fields):
                if skip_interp:
                    ret_dict[qoi] = state[..., i]
                else:
                    reshaped_states = state[..., i].reshape(-1, self.num_pts).T  # (num_pts, ...)
                    interp = self.interpolator()(grid_coords, reshaped_states, **self.interpolate_opts)
                    yp = interp(n_coords)
                    ret_dict[qoi] = yp.T.reshape(*loop_shape, *coords_shape)

            all_qois[j] = ret_dict

        if coords_obj_array:
            # Make an object array for each qoi, where each element is a unique `(*loop_shape, *coord_shape)` array
            for _, _first_dict in np.ndenumerate(all_qois):
                if _first_dict is not None:
                    break
            ret = {qoi: np.empty(all_qois.shape, dtype=object) for qoi in _first_dict}
            for qoi in ret:
                for index, qoi_dict in np.ndenumerate(all_qois):
                    if qoi_dict is not None:
                        ret[qoi][index] = qoi_dict[qoi]
        else:
            # Otherwise, all loop dims used the same coords, so just return the single array for each qoi
            ret = all_qois[0]

        return ret

    def interpolate_to_grid(self, field_coords: np.ndarray, field_values):
        """Interpolate the field values at given coordinates to the compression grid. An array of nan is returned
        for any coordinates or field values that are empty or None.

        :param field_coords: `(*coord_shape, dim)` - the coordinates of the field values; if an object array, then
                             each element is assumed to be a unique `(*coord_shape, dim)` array
        :param field_values: `dict` of `(*loop_shape, *coord_shape)` for each qoi - the field values at the coordinates;
                              if each array is an object array, then each element is assumed to be a unique
                             `(*loop_shape, *coord_shape)` array corresponding to the `field_coords`
        :return: `(*loop_shape, dof)` - the interpolated values on the compression grid
        """
        field_coords = self._correct_coords(field_coords)
        grid_coords = self._correct_coords(self.coords)

        # Loop over each set of coordinates and field values (multiple if they are object arrays)
        # If only one set of coords, then they are assumed the same for each set of field values
        coords_obj_array = np.issubdtype(field_coords.dtype, np.object_)
        fields_obj_array = np.issubdtype(next(iter(field_values.values())).dtype, np.object_)
        def _iterate_coords_and_fields():
            if coords_obj_array:
                for index, c in np.ndenumerate(field_coords):  # assumes same number of coords and field values
                    yield index, c, {qoi: field_values[qoi][index] for qoi in field_values}
            elif fields_obj_array:
                for index in np.ndindex(next(iter(field_values.values())).shape):
                    yield index, field_coords, {qoi: field_values[qoi][index] for qoi in field_values}
            else:
                yield (0,), field_coords, field_values  # assumes same coords for all field values

        if coords_obj_array:
            shape = field_coords.shape
        elif fields_obj_array:
            shape = next(iter(field_values.values())).shape
        else:
            shape = (1,)

        always_skip_interp = not coords_obj_array and np.array_equal(field_coords, grid_coords)  # must be exact match

        all_states = np.empty(shape, dtype=object)  # are you in good hands?

        for j, f_coords, f_values in _iterate_coords_and_fields():
            if f_coords is None or any([val is None for val in f_values.values()]):  # Skip empty samples
                continue

            skip_interp = always_skip_interp or np.array_equal(f_coords, grid_coords)  # exact even for floats

            coords_shape = f_coords.shape[:-1]
            loop_shape = next(iter(f_values.values())).shape[:-len(coords_shape)]
            states = np.empty((*loop_shape, self.num_pts, self.num_qoi))
            f_coords = f_coords.reshape(-1, self.dim)
            for i, qoi in enumerate(self.fields):
                field_vals = f_values[qoi].reshape((*loop_shape, -1))  # (..., Q)
                if skip_interp:
                    states[..., i] = field_vals
                else:
                    field_vals = field_vals.reshape((-1, field_vals.shape[-1])).T  # (Q, ...)
                    interp = self.interpolator()(f_coords, field_vals, **self.interpolate_opts)
                    yg = interp(grid_coords)
                    states[..., i] = yg.T.reshape(*loop_shape, self.num_pts)
            all_states[j] = states.reshape((*loop_shape, self.dof))

        # All fields now on the same dof grid, so stack them in same array
        state_shape = ()
        for index in np.ndindex(all_states.shape):
            if all_states[index] is not None:
                state_shape = all_states[index].shape
                break
        ret_states = np.empty(shape + state_shape)

        for index, arr in np.ndenumerate(all_states):
            ret_states[index] = arr if arr is not None else np.nan

        if not (coords_obj_array or fields_obj_array):
            ret_states = np.squeeze(ret_states, axis=0)  # artificial leading dim for non-object arrays

        return ret_states

    @abstractmethod
    def compute_map(self, **kwargs):
        """Compute and store the compression map. Must set the value of `coords` and `_is_computed`. Should
        use the same normalization as the parent `Variable` object.

        !!! Note
            You should pass any required data to `compute_map` with the assumption that the data will be used in the
            shape `(num_pts, num_qoi)` where `num_qoi` is the length of `fields` and `num_pts` is the length of
            `coords`. This is the shape that the compression map should be constructed in.
        """
        raise NotImplementedError

    @abstractmethod
    def compress(self, data: np.ndarray) -> np.ndarray:
        """Compress the data into a latent space.

        :param data: `(..., dof)` - the data to compress from full size of `dof`
        :return: `(..., rank)` - the compressed latent space data with size `rank`
        """
        raise NotImplementedError

    @abstractmethod
    def reconstruct(self, compressed: np.ndarray) -> np.ndarray:
        """Reconstruct the compressed data back into the full `dof` space.

        :param compressed: `(..., rank)` - the compressed data to reconstruct
        :return: `(..., dof)` - the reconstructed data with full `dof`
        """
        raise NotImplementedError

    @abstractmethod
    def latent_size(self) -> int:
        """Return the size of the latent space."""
        raise NotImplementedError

    @abstractmethod
    def estimate_latent_ranges(self) -> list[tuple[float, float]]:
        """Estimate the range of the latent space coefficients."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, spec: dict) -> Compression:
        """Construct a `Compression` object from a spec dictionary."""
        method = spec.pop('method', 'svd').lower()
        match method:
            case 'svd':
                return SVD(**spec)
            case other:
                raise NotImplementedError(f"Compression method '{other}' is not implemented.")


@dataclass
class SVD(Compression):
    """A Singular Value Decomposition (SVD) compression method. The SVD will be computed on initialization if the
    `data_matrix` is provided.

    :ivar data_matrix: `(dof, num_samples)` - the data matrix
    :ivar projection_matrix: `(dof, rank)` - the projection matrix
    :ivar rank: the rank of the SVD decomposition
    :ivar energy_tol: the energy tolerance of the SVD decomposition
    :ivar reconstruction_tol: the reconstruction error tolerance of the SVD decomposition
    """
    data_matrix: np.ndarray = None          # (dof, num_samples)
    projection_matrix: np.ndarray = None    # (dof, rank)
    rank: int = None
    energy_tol: float = None
    reconstruction_tol: float = None

    def __post_init__(self):
        """Compute the SVD if the data matrix is provided."""
        if (data_matrix := self.data_matrix) is not None:
            self.compute_map(data_matrix, rank=self.rank, energy_tol=self.energy_tol,
                             reconstruction_tol=self.reconstruction_tol)

    def compute_map(self, data_matrix: np.ndarray | dict, rank: int = None, energy_tol: float = None,
                    reconstruction_tol: float = None):
        """Compute the SVD compression map from the data matrix. Recall that `dof` is the total number of degrees of
        freedom, equal to the number of grid points `num_pts` times the number of quantities of interest `num_qoi`
        at each grid point.

        **Rank priority:** if `rank` is provided, then it will be used. Otherwise, if `reconstruction_tol` is provided,
        then the rank will be chosen to meet this reconstruction error level. Finally, if `energy_tol` is provided,
        then the rank will be chosen to meet this energy fraction level (sum of squared singular values).

        :param data_matrix: `(dof, num_samples)` - the data matrix. If passed in as a `dict`, then the data matrix
                            will be formed by concatenating the values of the `dict` along the last axis in the order
                            of the `fields` attribute and flattening the last two axes. This is useful for passing
                            in a dictionary of field values like `{field1: (num_samples, num_pts), field2: ...}`
                            which ensures consistency of shape with the compression `coords`.
        :param rank: the rank of the SVD decomposition
        :param energy_tol: the energy tolerance of the SVD decomposition (defaults to 0.95)
        :param reconstruction_tol: the reconstruction error tolerance of the SVD decomposition
        """
        if isinstance(data_matrix, dict):
            data_matrix = np.concatenate([data_matrix[field][..., np.newaxis] for field in self.fields], axis=-1)
            data_matrix = data_matrix.reshape(*data_matrix.shape[:-2], -1).T  # (dof, num_samples)

        nan_idx = np.any(np.isnan(data_matrix), axis=0)
        data_matrix = data_matrix[:, ~nan_idx]
        u, s, vt = np.linalg.svd(data_matrix)
        energy_frac = np.cumsum(s ** 2 / np.sum(s ** 2))
        if rank := (rank or self.rank):
            energy_tol = energy_frac[rank - 1]
            reconstruction_tol = relative_error(u[:, :rank] @ u[:, :rank].T @ data_matrix, data_matrix)
        elif reconstruction_tol := (reconstruction_tol or self.reconstruction_tol):
            rank = u.shape[1]
            for r in range(1, u.shape[1] + 1):
                if relative_error(u[:, :r] @ u[:, :r].T @ data_matrix, data_matrix) <= reconstruction_tol:
                    rank = r
                    break
            energy_tol = energy_frac[rank - 1]
        else:
            energy_tol = energy_tol or self.energy_tol or 0.95
            idx = int(np.where(energy_frac >= energy_tol)[0][0])
            rank = idx + 1
            reconstruction_tol = relative_error(u[:, :rank] @ u[:, :rank].T @ data_matrix, data_matrix)

        self.data_matrix = data_matrix
        self.rank = rank
        self.energy_tol = energy_tol
        self.reconstruction_tol = reconstruction_tol
        self.projection_matrix = u[:, :rank]  # (dof, rank)
        self._map_computed = True

    def compress(self, data):
        return np.squeeze(self.projection_matrix.T @ data[..., np.newaxis], axis=-1)

    def reconstruct(self, compressed):
        return np.squeeze(self.projection_matrix @ compressed[..., np.newaxis], axis=-1)

    def latent_size(self):
        return self.rank

    def estimate_latent_ranges(self):
        if self.map_exists:
            latent_data = self.compress(self.data_matrix.T)  # (rank, num_samples)
            latent_min = np.min(latent_data, axis=0)
            latent_max = np.max(latent_data, axis=0)
            return [(lmin, lmax) for lmin, lmax in zip(latent_min, latent_max)]
        else:
            return None
