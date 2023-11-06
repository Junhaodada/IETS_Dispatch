import numpy as np
import pandas as pd
from docplex.mp.model import Model
from docplex.mp.solution import SolveSolution
from typing import List, Tuple
from sklearn.cluster import KMeans


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


class V:
    """
    值函数
    (X,R)->V
    ---------------------------------------------------
    R_t = {E_TS,E_BS}
    E_TS [0,200]
    E_BS [0,80]
    ---------------------------------------------------
    X_t = {P_PG,H_CHP,P_CHP,a_BS_c,a_BS_d,P_BS_c,P_BS_d,
    a_TS_c,a_TS_d, H_TS_c,H_TS_d,P_PV}
    ---------------------------------------------------
    P_PG t时买入的电 [0,~)
    P_CHP t时CHP的电功输出 [0,~)
    H_CHP t时CHP的热能输出 [0,~)
    a_BS_d BS的放电状态 {0,1}
    a_BS_c BS的充电状态 {0,1}
    P_BS_d BS的放电量 [0,24]
    P_BS_c BS的充电量 [0,24]
    a_TS_d TS的放热状态 {0,1}
    a_TS_c TS的储热状态 {0,1}
    H_TS_d TS的放热量 [0,100]
    H_TS_c TS的储热量 [0,100]
    P_RES RES放电量 [0,W_set.P_RES[t]]
    P_EC 冷 [0,88.1]
    H_AC 冷 [0,168]
    # Q_EC
    # Q_AC
    # C_BS
    # C_EP
    # C_CHP
    # F_MT
    """

    def __init__(self):
        self.value_table = {
            # (E_TS, E_BS, t): np.random.uniform(0, 100)
            (E_TS, E_BS, t): 0
            for t in range(T)
            for E_TS in range(E_TS_MAX + 1)
            for E_BS in range(E_BS_MAX + 1)
        }
        self.r_count_table = {
            (E_TS, E_BS): 0
            for E_TS in range(E_TS_MAX + 1)
            for E_BS in range(E_BS_MAX + 1)
        }

    def get_value(self, r: R, t):
        return self.value_table[int(r.E_TS[t]), int(r.E_BS[t]), t]

    def get_value2(self, r: Tuple, t):
        return self.value_table[int(r[0]), int(r[1]), t]

    def set_value(self, r: R, t, var):
        self.value_table[int(r.E_TS[t]), int(r.E_BS[t]), t] = var

    def update_r_count(self, r: R, t):
        self.r_count_table[int(r.E_TS[t]), int(r.E_BS[t])] += 1

    def get_r_count(self, r: R, t):
        return self.r_count_table[int(r.E_TS[t]), int(r.E_BS[t])]

    def __str__(self):
        return f'value table'


# 创建值函数
# paramters
T = 24
N = 200
N2 = 5
SAMPLE_SIZE = 10  # 1000
E_TS_MAX = 200  # KWh
E_TS_MIN = 0  # KWh
E_BS_MAX = 80  # KWh
E_BS_MIN = 0  # KWh

v_table = V()


# TL_LOAD_RATIO = 0.98  # 热负载比例


# process data
def init_data(mode='w'):
    """
    input data

    Returns:
        R_sets, W_sets
    """
    R_sets: List[R] = []
    W_sets: List[W] = []
    for i in range(SAMPLE_SIZE):
        # R_set = R()
        # for t in range(T):
        #     np.random.seed(0)
        #     R_set.E_TS[t] = np.random.uniform(E_TS_MIN, E_TS_MAX)
        #     R_set.E_BS[t] = np.random.uniform(E_BS_MIN, E_BS_MAX)
        # R_sets.append(R_set)
        R_sets.append(R())
        # np.random.seed(1)
        R_sets[i].E_BS[0] = np.random.uniform(E_BS_MIN, E_BS_MAX)
        R_sets[i].E_TS[0] = np.random.uniform(E_TS_MIN, E_TS_MAX)
        # R_sets[i].E_TS[0] = 150
        # R_sets[i].E_BS[0] = 60
    W_data = pd.read_excel('./data/W_data.xlsx', sheet_name='Sheet1')
    LOAD_STD = 0.03
    PRICE_STD = 0.1
    RES_STD = 0.2
    for i in range(SAMPLE_SIZE):
        W_set = W()
        for t in range(T):
            W_set.E_PRICE[t] = 0 if W_data.loc[t, 'E_PRICE'] == 0 else W_data.loc[t, 'E_PRICE'] + np.random.normal(0,
                                                                                                                   PRICE_STD)
            W_set.P_EL[t] = W_data.loc[t, 'P_EL'] + np.random.normal(0,
                                                                     LOAD_STD)
            W_set.H_TL[t] = W_data.loc[t, 'H_TL'] + np.random.normal(0,
                                                                     LOAD_STD)
            W_set.C_TL[t] = 0 if W_data.loc[t, 'C_TL'] == 0 else W_data.loc[t, 'C_TL'] + np.random.normal(0,
                                                                                                          LOAD_STD)

            W_set.P_RES[t] = 0 if W_data.loc[t, 'P_RES'] == 0 else W_data.loc[t, 'P_RES'] + np.random.normal(0,
                                                                                                             RES_STD)
        # 导出场景数据
        if mode == 'w':
            df = pd.DataFrame(
                {'t': {t: t for t in range(T)}, 'E_PRICE': W_set.E_PRICE, 'P_EL': W_set.P_EL, 'H_TL': W_set.H_TL,
                 'C_TL': W_set.C_TL,
                 'P_RES': W_set.P_RES})
            df.to_excel(f'./data/W/W_{i}.xlsx', index=False)
        W_sets.append(W_set)

    return R_sets, W_sets


def to_df(obj: object, filename: str):
    df = pd.DataFrame()
    for name, value in obj.__dict__.items():
        df[name] = pd.Series(list(value.values()))
    df.to_excel(f'./data/sample/{filename}.xlsx', index=False)
    print(f'{filename}.xlsx创建成功!')
    return df


def cluster_data(R_sets: List[R], W_sets: List[W], CLUSTERS_NUM=5):
    R_dfs = []
    for i, obj in enumerate(R_sets):
        df = to_df(obj, f'R_{i}')
        R_dfs.append(df)

    W_dfs = []
    for i, obj in enumerate(W_sets):
        df = to_df(obj, f'W_{i}')
        W_dfs.append(df)

    # 聚类
    R_cluster = KMeans(n_clusters=CLUSTERS_NUM).fit_predict(pd.concat(R_dfs))
    W_cluster = KMeans(n_clusters=CLUSTERS_NUM).fit_predict(pd.concat(W_dfs))

    # 将聚类结果写出
    for i in range(CLUSTERS_NUM):
        tmp = pd.concat(R_dfs)[R_cluster == i]
        tmp.to_excel(f'./data/cluster/R_cluster_{i}.xlsx', index=False)

        tmp = pd.concat(W_dfs)[W_cluster == i]
        tmp.to_excel(f'./data/cluster/W_cluster_{i}.xlsx', index=False)


# MILP model
def solve_milp_all(R_set: R, W_set: W, mode='r', data_path=''):
    """
    solve MILP model

    Args:
        R_set: R set
        W_set: W set

    Returns:
        solve details

    """
    # 创建模型
    model = Model(name='IETS MILP MODEL')

    # 决策变量
    E_TS = {t: model.continuous_var(lb=E_TS_MIN, ub=E_TS_MAX, name=f'E_TS_{t}') for t in range(T)}
    E_BS = {t: model.continuous_var(lb=E_BS_MIN, ub=E_BS_MAX, name=f'E_BS_{t}') for t in range(T)}
    # 初始资源状态
    model.add_constraint(E_BS[0] == R_set.E_BS[0])
    model.add_constraint(E_TS[0] == R_set.E_TS[0])
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
    a_BS_c = {t: model.binary_var(name=f'a_BS_c_{t}') for t in range(T)}
    # BS成本
    C_BS = {t: model.continuous_var(name=f'C_BS_{t}') for t in range(T)}
    for t in range(1, T):
        # (2.1)
        eta_BS_c, eta_BS_d = 0.98, 0.98
        model.add_constraint(E_BS[t] == E_BS[t - 1] + eta_BS_c * P_BS_c[t] - P_BS_d[t] / eta_BS_d)
    for t in range(T):
        # (2.2)
        model.add_constraint(P_BS_c[t] <= a_BS_c[t] * P_BS_CN_MAX)
        # (2.3)
        model.add_constraint(P_BS_d[t] <= a_BS_d[t] * P_BS_DN_MAX)
        # (2.4)
        # 见E_BS定义
        # (2.5)
        model.add_constraint(a_BS_c[t] + a_BS_d[t] <= 1)
        # (2.6)
        beta_BS = 0.01
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
    for t in range(T):
        model.add_constraint(C_EP[t] == W_set.E_PRICE[t] * P_PG[t])

    # 2.4 CHP
    # (2.10)
    eta_mt = 0.3
    F_MT = {t: model.continuous_var(name=f'F_MT_{t}') for t in range(T)}
    P_CHP = {t: model.continuous_var(name=f'P_CHP_{t}') for t in range(T)}

    for t in range(T):
        model.add_constraint(F_MT[t] == P_CHP[t] / eta_mt)
    # (2.11)
    H_CHP = {t: model.continuous_var(name=f'H_CHP_{t}') for t in range(T)}
    eta_loss = 0.2
    eta_hr = 0.8
    for t in range(T):
        model.add_constraint(H_CHP[t] == P_CHP[t] * eta_hr * (1 - eta_mt - eta_loss) / eta_mt)
    # (2.12)
    eta_gas = 3.24
    H_GAS = 9.78
    C_CHP = {t: model.continuous_var(name=f'C_CHP_{t}') for t in range(T)}
    for t in range(T):
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
    P_EC = {t: model.continuous_var(lb=0, ub=P_EC_MAX, name=f'P_EC_{t}') for t in range(T)}
    H_AC = {t: model.continuous_var(lb=0, ub=P_EC_MAX, name=f'H_AC_{t}') for t in range(T)}
    Q_EC = {t: model.continuous_var(name=f'Q_EC_{t}') for t in range(T)}
    Q_AC = {t: model.continuous_var(name=f'Q_AC_{t}') for t in range(T)}

    for t in range(T):
        # (2.19)
        model.add_constraint(Q_EC[t] == P_EC[t] * a_EC)
        # (2.20)
        model.add_constraint(Q_AC[t] == H_AC[t] * a_AC)
        # (2.21)~(2.22)
        # 见P_EC和H_AC定义

    # 2.7 Energy balance
    for t in range(T):
        # power balance
        model.add_constraint(P_RES[t] + P_PG[t] + P_BS_d[t] + P_CHP[t] == P_EC[t] + P_BS_c[t] + W_set.P_EL[t])
        # heat balance
        eta_he = 0.98
        model.add_constraint(H_AC[t] + W_set.H_TL[t] / eta_he + H_TS_c[t] == H_TS_d[t] + H_CHP[t])
        # cool balance
        model.add_constraint(H_AC[t] * a_AC + P_EC[t] * a_EC == W_set.C_TL[t])

    # 目标函数
    obj_expr = model.sum(C_EP[t] + C_CHP[t] + C_BS[t] for t in range(T))

    # 求解
    model.minimize(obj_expr)
    # print('MILP求解结果如下:')
    solution: SolveSolution = model.solve()
    variable_data = []

    if solution:
        # print(solution.get_value('a_BS_c_12'))
        # print(model.iter_variables())
        print('objective value: ', solution.objective_value)
        # print('decision variables: ')
        # for var in model.iter_variables():
        #     print(f"{var}: {solution[var]}")

        variable_data.append({"Variable": 'Obj_Value', "Value": solution.objective_value})
        for var in model.iter_variables():
            variable_data.append({"Variable": var.name, "Value": solution[var]})
        # print(model.get_solve_details())
        if mode == 'w':
            df = pd.DataFrame(variable_data)
            df.to_excel(data_path, index=False)
            print('求解成功!')
            print(f'打开求解文件{data_path}查看结果')
    else:
        print(model.get_solve_details())
        print('求解失败!')
    return solution


def solve_milp_v(R_set: R, W_set: W, t):
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
    E_TS = model.continuous_var(lb=E_TS_MIN, ub=E_TS_MAX, name=f'E_TS_t')
    E_BS = model.continuous_var(lb=E_BS_MIN, ub=E_BS_MAX, name=f'E_BS_t')
    # 创建约束
    # 2.1 BS
    # 电池放电量
    P_BS_DN_MIN, P_BS_DN_MAX = 0, 24  # 0-24kw
    P_BS_d = model.continuous_var(lb=P_BS_DN_MIN, ub=P_BS_DN_MAX, name=f'P_BS_d_t')
    # 电池充电量
    P_BS_CN_MIN, P_BS_CN_MAX = 0, 24  # 0-24kw
    P_BS_c = model.continuous_var(lb=P_BS_CN_MIN, ub=P_BS_CN_MAX, name=f'P_BS_c_t')
    # 电池放电状态 1/0
    a_BS_d = model.binary_var(name=f'a_BS_d_t')
    # 电池充电状态 1/0
    a_BS_c = model.binary_var(name=f'a_BS_c_t')
    # (2.1)
    eta_BS_c, eta_BS_d = 0.98, 0.98
    model.add_constraint(E_BS == R_set.E_BS[t - 1] + eta_BS_c * P_BS_c - P_BS_d / eta_BS_d)
    # (2.2)
    model.add_constraint(P_BS_c <= a_BS_c * P_BS_CN_MAX)
    # (2.3)
    model.add_constraint(P_BS_d <= a_BS_d * P_BS_DN_MAX)
    # (2.4)
    # 见E_BS定义
    # (2.5)
    model.add_constraint(a_BS_c + a_BS_d <= 1)
    # (2.6)
    beta_BS = 0.01
    C_BS = model.continuous_var(name=f'C_BS_t')
    model.add_constraint(C_BS == beta_BS * (P_BS_c + P_BS_d))

    # 2.2 RES
    # (2.7)
    P_RES = model.continuous_var(name=f'P_RES_t')
    model.add_constraint(P_RES <= W_set.P_RES[t])

    # 2.3 PG
    # (2.9)
    P_PG = model.continuous_var(name=f'P_PG_t')  # t时买入的电
    C_EP = model.continuous_var(name=f'C_EP_t')
    model.add_constraint(C_EP == W_set.E_PRICE[t] * P_PG)

    # 2.4 CHP
    # (2.10)
    eta_mt = 0.3
    F_MT = model.continuous_var(name=f'F_MT_t')
    P_CHP = model.continuous_var(name=f'P_CHP_t')
    model.add_constraint(F_MT == P_CHP / eta_mt)
    # (2.11)
    H_CHP = model.continuous_var(name=f'H_CHP_t')
    eta_loss = 0.2
    eta_hr = 0.8
    model.add_constraint(H_CHP == P_CHP * eta_hr * (1 - eta_mt - eta_loss) / eta_mt)
    # (2.12)
    eta_gas = 3.24
    H_GAS = 9.78
    C_CHP = model.continuous_var(name=f'C_CHP_t')
    model.add_constraint(C_CHP == eta_gas * F_MT / H_GAS)

    # 2.5 TS
    eta_TS_d, eta_TS_c = 0.01, 0.98
    H_TS_DN_MIN, H_TS_DN_MAX = 0, 100
    H_TS_d = model.continuous_var(lb=H_TS_DN_MIN, ub=H_TS_DN_MAX, name=f'H_TS_d_t')
    # 电池充电量
    H_TS_CN_MIN, H_TS_CN_MAX = 0, 100
    H_TS_c = model.continuous_var(lb=H_TS_CN_MIN, ub=H_TS_CN_MAX, name=f'H_TS_c_t')

    # 储热器放热状态 1/0
    a_TS_d = model.binary_var(name=f'a_TS_d_t')
    # 储热器蓄热状态 1/0
    a_TS_c = model.binary_var(name=f'a_TS_c_t')
    # (2.13)
    model.add_constraint(E_TS == (1 - eta_TS_d) * R_set.E_TS[t - 1] - H_TS_d + eta_TS_c * H_TS_c)
    # (2.14)~(2.18)
    model.add_constraint(H_TS_c >= a_TS_c * H_TS_CN_MIN)
    model.add_constraint(H_TS_c >= a_TS_c * H_TS_CN_MIN)
    model.add_constraint(H_TS_d <= a_TS_d * H_TS_DN_MAX)
    model.add_constraint(H_TS_d <= a_TS_d * H_TS_DN_MAX)
    model.add_constraint(a_TS_c + a_TS_d <= 1)
    # (2.18)见E_TS定义

    # 2.6 Cool
    P_EC_MAX = 88.1
    H_AC_MAX = 168
    a_EC = 4
    a_AC = 0.7
    P_EC = model.continuous_var(ub=P_EC_MAX, name=f'P_EC_t')
    H_AC = model.continuous_var(ub=P_EC_MAX, name=f'H_AC_t')
    Q_EC = model.continuous_var(name=f'Q_EC_t')
    Q_AC = model.continuous_var(name=f'Q_AC_t')
    # (2.19)
    model.add_constraint(Q_EC == P_EC * a_EC)
    # (2.20)
    model.add_constraint(Q_AC == H_AC * a_AC)
    # (2.21)~(2.22)
    # 见P_EC和H_AC定义

    # 2.7 Energy balance
    # power balance
    model.add_constraint(P_RES + P_PG + P_BS_d + P_CHP == P_EC + P_BS_c + W_set.P_EL[t])
    # heat balance
    eta_he = 0.98
    model.add_constraint(H_AC + W_set.H_TL[t] / eta_he + H_TS_c == H_TS_d + H_CHP)
    # cool balance
    model.add_constraint(H_AC * a_AC + P_EC * a_EC == W_set.C_TL[t])

    # 引入γ_g
    # G = E_BS_MAX * E_TS_MAX
    # gama_g = {g: model.binary_var(name=f'gema_{g}') for g in range(G)}
    # model.add_constraint(model.sum(gama_g[g] for g in range(G)) == 1)

    # V_t_plus_1 = 0
    # i = 0
    # for E_TS in range(E_TS_MAX):
    #     for E_BS in range(E_BS_MAX):
    #         V_t_plus_1 += gama_g[i] * v_table.get_value2((E_TS, E_BS), t + 1)
    #         i += 1
    # 目标函数
    # obj_expr = C_EP + C_CHP + C_BS + V_t_plus_1
    obj_expr = C_EP + C_CHP + C_BS + v_table.get_value(R_set, t)

    # 求解
    model.minimize(obj_expr)
    # print('MILP求解结果如下:')
    solution: SolveSolution = model.solve()

    # 状态转移
    R_set.E_BS[t] = solution['E_BS_t']
    R_set.E_TS[t] = solution['E_TS_t']

    return solution


if __name__ == '__main__':
    R_sets, W_sets = init_data(mode='w')
    for i in range(SAMPLE_SIZE):
        solve_milp_all(R_sets[i], W_sets[i], mode='w', data_path=f'./data/Solution/Solution_{i}.xlsx')
