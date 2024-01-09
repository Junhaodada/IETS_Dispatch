"""
all parameters
"""
# init
T = 24  # Time period
N = 100
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
H_TS_D_MIN, H_TS_D_MAX = 0, 1300  # kW
H_TS_C_MIN, H_TS_C_MAX = 0, 1300  # kW
eta_TS, eta_TS_d, eta_TS_c = 0.05, 0.98, 0.98
# CHP
eta_chp_h, eta_chp_e = 0.35, 0.35
lambda_gas = 9.7
beta_chp = 0.018
P_CHP_MIN, P_CHP_MAX = 0, 1500  # kW
P_RD_MIN, P_RU_MAX = -300, 300  # kW
# Cool
H_AC_MIN, H_AC_MAX = 0, 2500  # kW
P_EC_MIN, P_EC_MAX = 0, 2500  # kW
a_ec, a_ac = 4, 1.2
