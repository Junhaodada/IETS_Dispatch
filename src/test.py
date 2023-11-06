import numpy as np

E_TS_MAX = 200  # KWh
E_BS_MAX = 80  # KWh

E_TS_SPACE = np.zeros((E_TS_MAX, E_BS_MAX))
for j in range(E_BS_MAX):
    E_TS_SPACE[:, j] = np.arange(0, E_TS_MAX)

E_BS_SPACE = np.zeros((E_TS_MAX, E_BS_MAX))
for j in range(E_TS_MAX):
    E_BS_SPACE[j, :] = np.arange(0, E_BS_MAX)

print(E_TS_SPACE)
print(E_BS_SPACE)
print([i for i in range(0, E_BS_MAX, 10)])