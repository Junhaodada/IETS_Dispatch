"""
all parameters
"""
# init
T = 24  # Time period
N = 10
N2 = 5
SAMPLE_SIZE = 5  # 1000

# milp
E_TS_MIN, E_TS_MAX = 800, 2500  # kWh
E_BS_MIN, E_BS_MAX = 300, 1000  # kWh
# BS
P_BS_D_MIN, P_BS_D_MAX = 0, 200  # kWh
P_BS_C_MIN, P_BS_C_MAX = 0, 200  # kWh
eta_BS, eta_BS_c, eta_BS_d = 0.05, 0.96, 0.96
beta_BS = 0.001
# TS
H_TS_D_MIN, H_TS_D_MAX = 0, 1300
H_TS_C_MIN, H_TS_C_MAX = 0, 1300
eta_TS, eta_TS_d, eta_TS_c = 0.05, 0.98, 0.98
# RES
P_PV_MIN, P_PV_MAX = 0, 10
eta_PV = 0.98  # !
# PG
P_PG_MIN, P_PG_MAX = 0, 10
# CHP
eta_mt = 0.35
eta_loss = 0.35
eta_hr = 0.35
