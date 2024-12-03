import numpy as np

coords = np.empty((4, 3), dtype=object)
field_values = np.empty((4, 3), dtype=object)
for index in np.ndindex(coords.shape):
    coords[index] = np.random.rand(2)
    field_values[index] = np.random.rand(5, 2)

ret_states = np.empty((*field_values.shape, 5, 2))

for index, arr in np.ndenumerate(field_values):
    ret_states[index] = arr
