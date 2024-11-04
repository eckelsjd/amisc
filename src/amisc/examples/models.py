"""Provides some example test functions for demonstration purposes."""
import pickle
import uuid
from pathlib import Path

import numpy as np

from amisc.system import System


def f1(x):
    """Figure 5 in Jakeman 2022."""
    y1 = x * np.sin(np.pi * x)
    return y1


def f2(y1):
    """Figure 5 in Jakeman 2022."""
    y2 = 1 / (1 + 25*y1**2)
    return y2


def f3(x, y2):
    """Hypothetical third function for Figure 5 in Jakeman 2022."""
    y3 = x * np.cos(np.pi * y2)
    return y3


def tanh_func(inputs, A=2, L=1, frac=4):
    """Simple tunable tanh function."""
    return {'y': A * np.tanh(2 / (L/frac) * (inputs['x'] - L/2)) + A + 1}


def borehole_func(inputs):
    """Model found at https://www.sfu.ca/~ssurjano/borehole.html

    :param inputs: `dict` of input variables `rw`, `r`, `Tu`, `Hu`, `Tl`, `Hl`, `L`, and `Kw`
    :returns vdot: Water flow rate in m^3/yr
    """
    rw = inputs['rw']   # Radius of borehole (m)
    r = inputs['r']     # Radius of influence (m)
    Tu = inputs['Tu']   # Transmissivity (m^2/yr)
    Hu = inputs['Hu']   # Potentiometric head (m)
    Tl = inputs['Tl']   # Transmissivity (m^2/yr)
    Hl = inputs['Hl']   # Potentiometric head (m)
    L = inputs['L']     # Length of borehole (m)
    Kw = inputs['Kw']   # Hydraulic conductivity (m/yr)
    vdot = 2*np.pi*Tu*(Hu-Hl) / (np.log(r/rw) * (1 + (2*L*Tu/(np.log(r/rw)*Kw*rw**2)) + (Tu/Tl)))

    return {'vdot': vdot}


def wing_weight_func(inputs):
    """Model found at https://www.sfu.ca/~ssurjano/wingweight.html

    :param inputs: `dict` of input variables `Sw`, `Wfw`, `A`, `Lambda`, `q`, `lamb`, `tc`, `Nz`, `Wdg`, and `Wp`
    :returns Wwing: the weight of the airplane wing (lb)
    """
    Sw = inputs['Sw']           # Wing area (ft^2)
    Wfw = inputs['Wfw']         # Weight of fuel (lb)
    A = inputs['A']             # Aspect ratio
    Lambda = inputs['Lambda']   # Quarter-chord sweep (deg)
    q = inputs['q']             # Dynamic pressure (lb/ft^2)
    lamb = inputs['lambda']     # taper ratio
    tc = inputs['tc']           # Aerofoil thickness to chord ratio
    Nz = inputs['Nz']           # Ultimate load factor
    Wdg = inputs['Wdg']         # Design gross weight (lb)
    Wp = inputs['Wp']           # Paint weight (lb/ft^2)
    Lambda = Lambda*(np.pi/180)
    Wwing = 0.036*(Sw**0.758)*(Wfw**0.0035)*((A/(np.cos(Lambda))**2)**0.6)*(q**0.006)*(lamb**0.04)*\
            (100*tc/np.cos(Lambda))**(-0.3)*((Nz*Wdg)**0.49) + Sw*Wp

    return {'Wwing': Wwing}


def nonlinear_wave(inputs, env_var=0.1**2, wavelength=0.5, wave_amp=0.1, tanh_amp=0.5, L=1, t=0.25):
    """Custom nonlinear model of a traveling Gaussian wave for testing.

    :param inputs: `dict` of input variables `d` and `theta`
    :param env_var: variance of Gaussian envelope
    :param wavelength: sinusoidal perturbation wavelength
    :param wave_amp: amplitude of perturbation
    :param tanh_amp: amplitude of tanh(x)
    :param L: domain length of underlying tanh function
    :param t: transition length of tanh function (as fraction of L)
    :returns: `dict` with model output `y`
    """
    d = inputs['d']
    theta = inputs['theta']

    # Traveling sinusoid with moving Gaussian envelope
    env_range = [0.2, 0.6]
    mu = env_range[0] + theta * (env_range[1] - env_range[0])
    theta_env = 1 / (np.sqrt(2 * np.pi * env_var)) * np.exp(-0.5 * (d - mu) ** 2 / env_var)
    ftheta = wave_amp * np.sin((2*np.pi/wavelength) * theta) * theta_env

    # Underlying tanh dependence on d
    fd = tanh_amp * np.tanh(2/(L*t)*(d - L/2)) + tanh_amp

    # Compute model = f(theta, d) + f(d)
    return {'y': ftheta + fd}


def fire_sat_globals():
    """Global variables for the fire satellite system."""
    Re = 6378140.    # Radius of Earth (m)
    mu = 3.986e14    # Gravitational parameter (m^3 s^-2)
    eta = 0.22       # Power efficiency
    Id = 0.77        # Inherent degradation of the array
    thetai = 0.      # Sun incidence angle
    LT = 15.         # Spacecraft lifetime (years)
    eps = 0.0375     # Power production degradation (%/year)
    rlw = 3.         # Length to width ratio
    nsa = 3.         # Number of solar arrays
    rho = 700.       # Mass density of arrays (kg/m^3)
    t = 0.005        # Thickness (m)
    D = 2.           # Distance between panels (m)
    IbodyX = 6200.   # kg*m^2
    IbodyY = 6200.   # kg*m^2
    IbodyZ = 4700.   # kg*m^2
    dt_slew = 760.   # s
    theta = 15.      # Deviation of moment axis from vertical (deg)
    As = 13.85       # Area reflecting radiation (m^2)
    c = 2.9979e8     # Speed of light (m/s)
    M = 7.96e15      # Magnetic moment of earth (A*m^2)
    Rd = 5.          # Residual dipole of spacecraft (A*m^2)
    rhoa=5.148e-11   # Atmospheric density (kg/m^3) -- typo in Chaudhuri 2018 has this as 1e11 instead
    A = 13.85        # Cross-section in flight (m^2)
    Phold = 20.      # Holding power (W)
    omega = 6000.    # Max vel of wheel (rpm)
    nrw = 3.         # Number of reaction wheels

    return {k: v for k, v in locals().items() if not k.startswith('_') and isinstance(v, float)}


def orbit_fun(inputs, output_path=None, pct_failure=0):
    """Compute the orbit model for the fire satellite system.

    :param inputs: `dict` of input variables `H` and `Φ`
    :param output_path: where to save model outputs
    :param pct_failure: probability of a failure
    :returns: `dict` with model outputs `Vsat`, `To`, `Te`, and `Slew`
    """
    v = fire_sat_globals()                              # Global variables

    H = inputs['H']                                     # Altitude (m)
    phi = inputs[u'Φ']                                  # Target diameter (m)
    vel = np.sqrt(v['mu'] / (v['Re'] + H))              # Satellite velocity (m/s)
    dt_orbit = 2*np.pi*(v['Re'] + H) / vel                                                     # Orbit period (s)
    dt_eclipse = (dt_orbit/np.pi)*np.arcsin(v['Re'] / (v['Re'] + H))                           # Eclipse period (s)
    theta_slew = np.arctan(np.sin(phi / v['Re']) / (1 - np.cos(phi / v['Re']) + H/v['Re']))    # Max slew angle (rad)

    num_samples = np.atleast_1d(H).shape[0]

    if np.random.rand() < pct_failure:
        i = np.random.randint(0, num_samples)
        i2 = np.random.randint(0, num_samples)
        vel[i] = np.nan
        theta_slew[i2] = np.nan

    y = {'Vsat': vel, 'To': dt_orbit, 'Te': dt_eclipse, 'Slew': theta_slew}

    if output_path is not None:
        files = []
        id = str(uuid.uuid4())
        for index in range(num_samples):
            fname = f'{id}_{index}.pkl'
            with open(Path(output_path) / fname, 'wb') as fd:
                pickle.dump({var: np.atleast_1d(y[var])[index] for var in y.keys()}, fd)
            files.append(fname)
        y['output_path'] = files

    return y


def power_fun(inputs, model_fidelity=(0,), *, output_path=None, pct_failure=0):
    """Compute the power model for the fire satellite system.

    :param inputs: `dict` of input variables `Po`, `Fs`, `To`, `Te`, and `Pat`
    :param model_fidelity: fidelity index for the model
    :param output_path: where to save model outputs
    :param pct_failure: probability of a failure
    :returns: `dict` with model outputs `Imin`, `Imax`, `Ptot`, and `Asa`
    """
    alpha = model_fidelity
    pct = 1 - (2 - alpha[0]) * 0.04 if len(alpha) == 1 else 1  # extra pct error term
    v = fire_sat_globals()       # Global variables

    Po = inputs['Po']            # Other power sources (W)
    Fs = inputs['Fs']            # Solar flux (W/m^2)
    dt_orbit = inputs['To']      # Orbit period (s)
    dt_eclipse = inputs['Te']    # Eclipse period (s)
    Pacs = inputs['Pat']         # Power from attitude control system (W)

    Ptot = Po + Pacs
    Pe = Ptot
    Pd = Ptot
    Xe = 0.6                     # These are power efficiencies in eclipse and daylight
    Xd = 0.8                     # See Ch. 11 of Wertz 1999 SMAD
    Te = dt_eclipse
    Td = dt_orbit - Te
    Psa = ((Pe*Te/Xe) + (Pd*Td/Xd)) / Td
    Pbol = v['eta'] * Fs * v['Id'] * np.cos(v['thetai'])
    Peol = Pbol * (1 - v['eps']) ** v['LT']
    Asa = Psa / Peol                                # Total solar array size (m^2)
    L = np.sqrt(Asa * v['rlw'] / v['nsa'])
    W = np.sqrt(Asa / (v['rlw'] * v['nsa']))
    msa = 2 * v['rho'] * L * W * v['t']              # Mass of solar array
    Ix = msa*((1/12)*(L**2 + v['t']**2) + (v['D']+L/2)**2) + v['IbodyX']
    Iy = (msa/12)*(L**2 + W**2) + v['IbodyY']        # typo in Zaman 2013 has this as H**2 instead of L**2
    Iz = msa*((1/12)*(L**2 + W**2) + (v['D'] + L/2)**2) + v['IbodyZ']
    Itot = np.concatenate((Ix[..., np.newaxis], Iy[..., np.newaxis], Iz[..., np.newaxis]), axis=-1)
    Imin = np.min(Itot, axis=-1)
    Imax = np.max(Itot, axis=-1)

    num_samples = np.atleast_1d(Po).shape[0]

    if np.random.rand() < pct_failure:
        i = np.random.randint(0, num_samples)
        i2 = np.random.randint(0, num_samples)
        Imin[i2] = np.nan
        Asa[i] = np.nan

    y = {'Imin': Imin, 'Imax': Imax*pct, 'Ptot': Ptot*pct, 'Asa': Asa}

    if output_path is not None:
        files = []
        id = str(uuid.uuid4())
        for index in range(num_samples):
            fname = f'{id}_{index}.pkl'
            with open(Path(output_path) / fname, 'wb') as fd:
                pickle.dump({var: np.atleast_1d(y[var])[index] for var in y.keys()}, fd)
            files.append(fname)
        y['output_path'] = files

    return y


def attitude_fun(inputs, model_fidelity=(0,)):
    """Compute the attitude model for the fire satellite system.

    :param inputs: `dict` of input variables `H`, `Fs`, `Lsp`, `q`, `La`, `Cd`, `Vsat`, and `Slew`
    :param model_fidelity: fidelity index for the model
    :returns: `dict` with model outputs `Pat` and `tau_tot`
    """
    alpha = model_fidelity
    v = fire_sat_globals()       # Global variables
    pct = 1 - (2 - alpha[0])*0.04 if len(alpha) == 1 else 1  # extra model fidelity pct error term

    H = inputs['H']              # Altitude (m)
    Fs = inputs['Fs']            # Solar flux
    Lsp = inputs['Lsp']          # Moment arm for solar radiation pressure
    q = inputs['q']              # Reflectance factor
    La = inputs['La']            # Moment arm for aerodynamic drag
    Cd = inputs['Cd']            # Drag coefficient
    vel = inputs['Vsat']         # Satellite velocity
    theta_slew = inputs['Slew']  # Max slew angle
    Imin = inputs['Imin']        # Minimum moment of inertia
    Imax = inputs['Imax']        # Maximum moment of inertia

    tau_slew = 4*theta_slew*Imax / v['dt_slew']**2
    tau_g = 3 * v['mu'] * np.abs(Imax - Imin) * np.sin(2 * v['theta'] * (np.pi / 180)) / (2 * (v['Re'] + H) ** 3)
    tau_sp = Lsp * Fs * v['As'] * (1 + q) * np.cos(v['thetai']) / v['c']
    tau_m = 2 * v['M'] * v['Rd'] / (v['Re'] + H) ** 3
    tau_a = (1 / 2) * La * v['rhoa'] * Cd * v['A'] * vel ** 2
    tau_dist = np.sqrt(tau_g**2 + tau_sp**2 + tau_m**2 + tau_a**2)
    tau_tot = np.max(np.concatenate((tau_slew[..., np.newaxis], tau_dist[..., np.newaxis]), axis=-1), axis=-1)
    Pacs = tau_tot * (v['omega'] * (2 * np.pi / 60)) + v['nrw'] * v['Phold']

    y = {'Pat': Pacs*pct, 'tau_tot': tau_tot*pct}

    return y


def fire_sat_system(save_dir=None):
    """Fire satellite detection system model from Chaudhuri 2018.

    !!! Note "Some modifications"
        Orbit will save outputs all the time; Power won't because it is part of an FPI loop. Orbit and Power can
        return `np.nan` some of the time (optional, to test robustness of surrogate). Power and Attitude have an
        `alpha` fidelity index that controls accuracy of the returned values. `alpha=0,1,2` corresponds to `8,4,0`
        percent error.

    :param save_dir: where to save model outputs
    :returns: a `System` object for the fire sat MD example system
    """
    return System.load_from_file(Path(__file__).parent / 'fire-sat.yml', root_dir=save_dir)
