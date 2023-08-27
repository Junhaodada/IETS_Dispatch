import numpy as np
import pandas as pd
from docplex.mp.model import Model
from docplex.mp.solution import SolveSolution
from typing import List

class W:
    """
    Exogenous information `W_t` of IETS includes stochastic processes of RES generation,
    ambient temperature,electrical and thermal loads, and real-time price.
    """

    def __init__(self):
        self.E_PRICE = {t: 0 for t in range(T)}  # electrical real-time price
        self.P_EL = {t: 0 for t in range(T)}  # active electrical load
        self.H_TL = {t: 0 for t in range(T)}  # thermal load in node
        self.C_TL = {t: 0 for t in range(T)}  # cool load in node
        # self.TEM_AM = {t: 0 for t in range(T)}  # ambient temperature
        self.P_RES = {t: 0 for t in range(T)}  # Upper active power output limit of RES


class R:
    """
    resource state `Rt` includes E_TS and E_BS (Energy stored in TS/BS at time-slot t)
    """

    def __init__(self) -> None:
        self.E_TS = {t: 0 for t in range(T)}  # 0-200 kWh
        self.E_BS = {t: 0 for t in range(T)}  # 0-80 kWh


# paramters
T = 24
SAMPLE_SIZE = 1  # 1000
E_TS_MAX = 200  # KWh
E_TS_MIN = 0  # KWh
E_BS_MAX = 80  # KWh
E_BS_MIN = 0  # KWh
# TL_LOAD_RATIO = 0.98  # 热负载比例


def init_data():
    """
    input data

    Returns:
        R_sets, W_sets
    """
    R_sets: List[R] = []
    W_sets: List[W] = []
    for i in range(SAMPLE_SIZE):
        R_set = R()
        for t in range(T):
            np.random.seed(0)
            R_set.E_TS[t] = np.random.uniform(E_TS_MIN, E_TS_MAX)
            R_set.E_BS[t] = np.random.uniform(E_BS_MIN, E_BS_MAX)
        R_sets.append(R_set)
    W_data = pd.read_excel('./data/W_data.xlsx', sheet_name='Sheet1')
    for i in range(SAMPLE_SIZE):
        W_set = W()
        for t in range(T):
            W_set.E_PRICE[t] = W_data.loc[t, 'E_PRICE']
            # W_set.TEM_AM[t] = W_data.loc[t, 'TEM_AM']
            W_set.P_EL[t] = W_data.loc[t, 'P_EL']
            W_set.H_TL[t] = W_data.loc[t, 'H_TL']
            W_set.C_TL[t] = W_data.loc[t, 'C_TL']
            W_set.P_RES[t] = W_data.loc[t, 'P_RES']
        W_sets.append(W_set)
    return R_sets, W_sets


# MILP model
def solve_milp(R_set: R, W_set: W, t):
    """
    solve MILP model

    Args:
        R_set: R set
        W_set: W set
        t: t plot

    Returns:
        solve details

    """
    # 创建模型
    model = Model(name='IETS MILP MODEL')

    # 决策变量
    E_TS = {t: model.continuous_var(lb=E_TS_MIN, ub=E_TS_MAX, name=f'E_TS_{t}') for t in range(T)}
    E_BS = {t: model.continuous_var(lb=E_BS_MIN, ub=E_BS_MAX, name=f'E_BS_{t}') for t in range(T)}
    model.add_constraint(E_TS[0] == R_set.E_TS[0])
    model.add_constraint(E_BS[0] == R_set.E_BS[0])

    # 创建约束
    # 2.1 BS
    # 电池放电量
    P_BS_DN_MIN, P_BS_DN_MAX = 0, 24  # 0-24kw
    P_BS_d = {t: model.continuous_var(lb=P_BS_DN_MIN, ub=P_BS_DN_MAX, name=f'P_BS_d_{t}') for t in range(T)}
    # 电池充电量
    P_BS_CN_MIN, P_BS_CN_MAX = 0, 24  # 0-24kw
    P_BS_c = {t: model.continuous_var(lb=P_BS_CN_MIN, ub=P_BS_CN_MAX, name=f'P_BS_c_{t}') for t in range(T)}
    # 电池放电状态 1/0
    a_BS_d = {t: model.binary_var(name=f'a_BS_d_{t}') for t in range(T)}
    # 电池充电状态 1/0
    a_BS_c = {t: model.binary_var(name=f'a_BS_c_{t}') for t in range(T)}
    # (2.1)
    eta_BS_c, eta_BS_d = 0.98, 0.98
    for t in range(1, T):
        model.add_constraint(E_BS[t] == E_BS[t - 1] + eta_BS_c * P_BS_c[t] - P_BS_d[t] / eta_BS_d)
    # (2.2)
    for t in range(T):
        model.add_constraint(P_BS_c[t] <= a_BS_c[t] * P_BS_CN_MAX)
    # (2.3)
    for t in range(T):
        model.add_constraint(P_BS_d[t] <= a_BS_d[t] * P_BS_DN_MAX)
    # (2.4)
    # 见E_BS定义
    # (2.5)
    for t in range(T):
        model.add_constraint(a_BS_c[t] + a_BS_d[t] <= 1)
    # (2.6)
    beta_BS = 0.01
    C_BS = {t: model.continuous_var(name=f'C_BS_{t}') for t in range(T)}
    for t in range(T):
        model.add_constraint(C_BS[t] == beta_BS * (P_BS_c[t] + P_BS_d[t]))

    # 2.2 RES
    # (2.7)
    P_RES = {t: model.continuous_var(name=f'P_RES_{t}') for t in range(T)}
    for t in range(T):
        model.add_constraint(P_RES[t] <= W_set.P_RES[t])

    # 2.3 PG
    # (2.9)
    P_PG = {t: model.continuous_var(name=f'P_PG_{t}') for t in range(T)}  # t时买入的电
    C_EP = {t: model.continuous_var(name=f'C_EP_{t}') for t in range(T)}
    model.add_constraint(C_EP[t] == W_set.E_PRICE[t] * P_PG[t])

    # 2.4 CHP
    # (2.10)
    eta_mt = 0.3
    F_MT = {t: model.continuous_var(name=f'F_MT_{t}') for t in range(T)}
    P_CHP = {t: model.continuous_var(name=f'P_CHP_{t}') for t in range(T)}
    model.add_constraint(F_MT[t] == P_CHP[t] / eta_mt)
    # (2.11)
    H_CHP = {t: model.continuous_var(name=f'H_CHP_{t}') for t in range(T)}
    eta_loss = 0.2
    eta_hr = 0.8
    model.add_constraint(H_CHP[t] == P_CHP[t] * eta_hr * (1 - eta_mt - eta_loss) / eta_mt)
    # (2.12)
    eta_gas = 3.24
    H_GAS = 9.78
    C_CHP = {t: model.continuous_var(name=f'C_CHP_{t}') for t in range(T)}
    model.add_constraint(C_CHP[t] == eta_gas * F_MT[t] / H_GAS)

    # 2.5 TS
    eta_TS_d, eta_TS_c = 0.01, 0.98
    H_TS_DN_MIN, H_TS_DN_MAX = 0, 100
    H_TS_d = {t: model.continuous_var(lb=H_TS_DN_MIN, ub=H_TS_DN_MAX, name=f'H_TS_d_{t}') for t in range(T)}
    # 电池充电量
    H_TS_CN_MIN, H_TS_CN_MAX = 0, 100
    H_TS_c = {t: model.continuous_var(lb=H_TS_CN_MIN, ub=H_TS_CN_MAX, name=f'H_TS_c_{t}') for t in range(T)}

    # 储热器放热状态 1/0
    a_TS_d = {t: model.binary_var(name=f'a_TS_d_{t}') for t in range(T)}
    # a_TS_d = {t: model.integer_var(lb=-1,ub=0,name=f'a_TS_d_{t}') for t in range(T)}
    # 储热器蓄热状态 1/0
    a_TS_c = {t: model.binary_var(name=f'a_TS_c_{t}') for t in range(T)}
    for t in range(1, T):
        # (2.13)
        model.add_constraint(E_TS[t] == (1 - eta_TS_d) * E_TS[t - 1] - H_TS_d[t] + eta_TS_c * H_TS_c[t])
    for t in range(T):
        # (2.14)~(2.18)
        model.add_constraint(H_TS_c[t] >= a_TS_c[t] * H_TS_CN_MIN)
        model.add_constraint(H_TS_c[t] >= a_TS_c[t] * H_TS_CN_MIN)
        model.add_constraint(H_TS_d[t] <= a_TS_d[t] * H_TS_DN_MAX)
        model.add_constraint(H_TS_d[t] <= a_TS_d[t] * H_TS_DN_MAX)
        model.add_constraint(a_TS_c[t] + a_TS_d[t] <= 1)
        # (2.18)见E_TS定义

    # 2.6 Cool
    P_EC_MAX = 88.1
    H_AC_MAX = 168
    a_EC = 4
    a_AC = 0.7
    P_EC = {t: model.continuous_var(ub=P_EC_MAX, name=f'P_EC_{t}') for t in range(T)}
    H_AC = {t: model.continuous_var(ub=P_EC_MAX, name=f'H_AC_{t}') for t in range(T)}
    Q_EC = {t: model.continuous_var(name=f'Q_EC_{t}') for t in range(T)}
    Q_AC = {t: model.continuous_var(name=f'Q_AC_{t}') for t in range(T)}
    # (2.19)
    model.add_constraint(Q_EC[t] == P_EC[t] * a_EC)
    # (2.20)
    model.add_constraint(Q_AC[t] == H_AC[t] * a_AC)
    # (2.21)~(2.22)
    # 见P_EC和H_AC定义

    # 2.7 Energy balance
    # power balance
    model.add_constraint(P_RES[t] + P_PG[t] + P_BS_d[t] + P_CHP[t] == P_EC[t] + P_BS_c[t] + W_set.P_EL[t])
    # heat balance
    eta_he = 0.98
    model.add_constraint(H_AC[t] + W_set.H_TL[t] / eta_he + H_TS_c[t] == H_TS_d[t] + H_CHP[t])
    # cool balance
    model.add_constraint(H_AC[t] * a_AC + P_EC[t] * a_EC == W_set.C_TL[t])

    # 目标函数
    obj_expr = C_EP[t] + C_CHP[t] + C_BS[t]+C_CHP[t]

    # 求解
    model.minimize(obj_expr)
    print('MILP求解结果如下:')
    solution:SolveSolution = model.solve()
    if solution:
        # print(solution.get_value('a_BS_c_12'))
        # print(model.iter_variables())
        print('objective value: ',solution.objective_value)
        print('decision variables: ')
        for var in model.iter_variables():
            print(f"{var}: {solution[var]}")
        variable_data = []

        for var in model.iter_variables():
            variable_data.append({"Variable": var, "Value": solution[var]})

        df = pd.DataFrame(variable_data)
        df.to_excel("./data/Solution.xlsx", index=False)
        # print(model.get_solve_details())
    else:
        print(model.get_solve_details())
    return solution


if __name__ == '__main__':
    R_sets, W_sets = init_data()
    print('测试模拟数据:')
    print(R_sets[0].E_BS[0])
    print(R_sets[0].E_TS[0])
    print(W_sets[0].H_TL[0])
    print(W_sets[0].H_TL[0])
    solve_milp(R_sets[0], W_sets[0], 2)
