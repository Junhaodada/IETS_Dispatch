"""
solve milp
"""
import numpy as np
import pandas as pd
from docplex.mp.model import Model
from docplex.mp.solution import SolveSolution
from typing import List, Tuple
from params import *


class W:
    """
    Exogenous information `W_t` of IETS includes stochastic processes of RES generation,
    ambient temperature,electrical and thermal loads, and real-time price.
    """

    def __init__(self):
        self.E_PRICE = {t: 0 for t in range(T)}  # electrical real-time price
        self.L_Power = {t: 0 for t in range(T)}  # active electrical load
        self.L_Heat = {t: 0 for t in range(T)}  # thermal load in node
        self.L_Cool = {t: 0 for t in range(T)}  # cool load in node
        # Upper active power output limit of RES
        self.P_PV = {t: 0 for t in range(T)}


class R:
    """
    resource state `Rt` includes E_TS and E_BS (Energy stored in TS/BS at time-slot t)
    """

    def __init__(self):
        self.E_TS = {t: 0 for t in range(T)}
        self.E_BS = {t: 0 for t in range(T)}


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
    P_PG t 时买入的电 [0,~)
    P_CHP t 时 CHP 的电功输出 [0,~)
    H_CHP t 时 CHP 的热能输出 [0,~)
    a_BS_d BS 的放电状态 {0,1}
    a_BS_c BS 的充电状态 {0,1}
    P_BS_d BS 的放电量 [0,24]
    P_BS_c BS 的充电量 [0,24]
    a_TS_d TS 的放热状态 {0,1}
    a_TS_c TS 的储热状态 {0,1}
    H_TS_d TS 的放热量 [0,100]
    H_TS_c TS 的储热量 [0,100]
    P_PV RES 放电量 [0,W_set.P_PV[t]]
    P_ec 冷 [0,88.1]
    H_ac 冷 [0,168]
    # C_ec
    # C_ac
    # C_BS
    # C_PG
    # C_CHP
    # F_gas
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

    def to_csv(self):
        """
        导出为 csv 格式
        """
        # 转换字典为 DataFrame
        df = pd.DataFrame(list(self.value_table.items()),
                          columns=['(E_TS, E_BS, t)', 'Value'])

        # 拆分 '(E_TS, E_BS, t)' 列为三列
        df[['E_TS', 'E_BS', 't']] = pd.DataFrame(
            df['(E_TS, E_BS, t)'].tolist(), index=df.index)
        df.drop(['(E_TS, E_BS, t)'], axis=1, inplace=True)

        # 使用 pivot 重新排列 DataFrame
        df_pivot = df.pivot(index='t', columns=[
                            'E_TS', 'E_BS'], values='Value')

        # 将 DataFrame 保存为 CSV 文件
        df_pivot.to_csv('data/v_table.csv')
        print('value_table save successfully!')

    def __str__(self):
        return f'value table'


v_table = V()


def init_data(mode='r'):
    """
    input data

    Args:
        mode: r or w

    Returns:
        R_sets, W_sets
    """
    LOAD_STD = 0.03
    PRICE_STD = 0.1
    RES_STD = 0.2
    R_sets: List[R] = []
    W_sets: List[W] = []
    for i in range(SAMPLE_SIZE):
        R_sets.append(R())
        np.random.seed(10)
        # R_sets[i].E_BS[0] = np.random.uniform(E_BS_MIN, E_BS_MAX)
        # R_sets[i].E_TS[0] = np.random.uniform(E_TS_MIN, E_TS_MAX)
        R_sets[i].E_TS[0] = 1000
        R_sets[i].E_BS[0] = 500
    W_data = pd.read_excel('./data/W_data.xlsx', sheet_name='Sheet1')

    for i in range(SAMPLE_SIZE):
        W_set = W()
        for t in range(T):
            # W_set.E_PRICE[t] = W_data.loc[t, 'E_PRICE'] + np.random.normal(0,
            #                                                                PRICE_STD)
            # W_set.L_Power[t] = W_data.loc[t, 'L_Power'] + np.random.normal(0,
            #                                                                LOAD_STD)
            # W_set.L_Heat[t] = W_data.loc[t, 'L_Heat'] + np.random.normal(0,
            #                                                              LOAD_STD)
            # W_set.L_Cool[t] = W_data.loc[t, 'L_Cool'] + np.random.normal(0,
            #                                                              LOAD_STD)
            # W_set.P_PV[t] = W_data.loc[t, 'P_PV'] + np.random.normal(0,
            #                                                          RES_STD)
            W_set.E_PRICE[t] = W_data.loc[t, 'E_PRICE']
            W_set.L_Power[t] = W_data.loc[t, 'L_Power']
            W_set.L_Heat[t] = W_data.loc[t, 'L_Heat']
            W_set.L_Cool[t] = W_data.loc[t, 'L_Cool']
            W_set.P_PV[t] = W_data.loc[t, 'P_PV']
        # 导出场景数据
        if mode == 'w':
            df = pd.DataFrame(
                {'t': {t: t for t in range(T)}, 'E_PRICE': W_set.E_PRICE, 'L_Power': W_set.L_Power, 'L_Heat': W_set.L_Heat,
                 'L_Cool': W_set.L_Cool,
                 'P_PV': W_set.P_PV})
            df.to_excel(f'./data/W/W_{i}.xlsx', index=False)
        W_sets.append(W_set)

    return R_sets, W_sets


def solve_milp_all(R_set: R, W_set: W, mode='r', data_path=''):
    """
    solve MILP model (known exogenous information)

    Args:
        R_set: R set
        W_set: W set
        mode: r or w
        data_path: ''

    Returns:
        solve details

    """
    # 创建模型
    model = Model(name='IETS MILP MODEL')

    # 决策变量
    E_TS = {t: model.continuous_var(
        lb=E_TS_MIN, ub=E_TS_MAX, name=f'E_TS_{t}') for t in range(T)}
    E_BS = {t: model.continuous_var(
        lb=E_BS_MIN, ub=E_BS_MAX, name=f'E_BS_{t}') for t in range(T)}

    # 初始资源状态
    model.add_constraint(E_BS[0] == R_set.E_BS[0])
    model.add_constraint(E_TS[0] == R_set.E_TS[0])

    # 创建约束
    # 2.1 BS
    # 电池放电量
    P_BS_d = {t: model.continuous_var(
        lb=P_BS_D_MIN, ub=P_BS_D_MAX, name=f'P_BS_d_{t}') for t in range(T)}
    # 电池充电量
    P_BS_c = {t: model.continuous_var(
        lb=P_BS_C_MIN, ub=P_BS_C_MAX, name=f'P_BS_c_{t}') for t in range(T)}
    # 电池放电状态 1/0
    a_BS_d = {t: model.binary_var(name=f'a_BS_d_{t}') for t in range(T)}
    a_BS_c = {t: model.binary_var(name=f'a_BS_c_{t}') for t in range(T)}
    # BS 成本
    C_BS = {t: model.continuous_var(name=f'C_BS_{t}') for t in range(T)}
    for t in range(T):
        # (2.1)
        model.add_constraint(P_BS_c[t] <= a_BS_c[t] * P_BS_C_MAX)
        # (2.2)
        model.add_constraint(P_BS_d[t] <= a_BS_d[t] * P_BS_D_MAX)
        # (2.3)
        model.add_constraint(a_BS_d[t] + a_BS_c[t] <= 1)
        # (2.4)
        # 略
    for t in range(1, T):
        # (2.5)
        model.add_constraint(E_BS[t] == (1 - eta_BS) * E_BS[t - 1] +
                             eta_BS_c * P_BS_c[t] - P_BS_d[t] / eta_BS_d)
    for t in range(T):
        # (2.6)
        # 见 E_BS 定义
        # (2.7)
        model.add_constraint(C_BS[t] == beta_BS * (P_BS_c[t] + P_BS_d[t]))

    # 2.2 TS
    # TS 放热量
    H_TS_d = {t: model.continuous_var(
        lb=H_TS_D_MIN, ub=H_TS_D_MAX, name=f'H_TS_d_{t}') for t in range(T)}
    # TS 储热量
    H_TS_c = {t: model.continuous_var(
        lb=H_TS_C_MIN, ub=H_TS_C_MAX, name=f'H_TS_c_{t}') for t in range(T)}
    # 储热器放热状态 1/0
    a_TS_d = {t: model.binary_var(name=f'a_TS_d_{t}') for t in range(T)}
    # 储热器蓄热状态 1/0
    a_TS_c = {t: model.binary_var(name=f'a_TS_c_{t}') for t in range(T)}
    for t in range(T):
        # (2.8)
        model.add_constraint(H_TS_c[t] >= a_TS_c[t] * H_TS_C_MIN)
        model.add_constraint(H_TS_c[t] <= a_TS_c[t] * H_TS_C_MAX)
        # (2.9)
        model.add_constraint(H_TS_d[t] >= a_TS_d[t] * H_TS_D_MIN)
        model.add_constraint(H_TS_d[t] <= a_TS_d[t] * H_TS_D_MAX)
        # (2.10)
        model.add_constraint(a_TS_c[t] + a_TS_d[t] <= 1)
        # (2.11)
        # 略
    for t in range(1, T):
        # (2.12)
        model.add_constraint(E_TS[t] == (
            1 - eta_TS) * E_TS[t - 1] + (eta_TS_c * H_TS_c[t] - H_TS_d[t] / eta_TS_d))
        # (2.13)
        # 见 E_TS 定义

    # 2.3 RES
    # 光伏输出电量
    P_PV = {t: model.continuous_var(name=f'P_PV_{t}') for t in range(T)}
    for t in range(T):
        # (2.14)
        # 略
        # (2.15)
        model.add_constraint(P_PV[t] <= W_set.P_PV[t])

    # 2.4 PG
    # 从电网购电量
    P_PG = {t: model.continuous_var(name=f'P_PG_{t}')
            for t in range(T)}  # t 时买入的电
    # 购电成本
    C_PG = {t: model.continuous_var(name=f'C_PG_{t}') for t in range(T)}
    for t in range(T):
        # (2.16)
        model.add_constraint(C_PG[t] == W_set.E_PRICE[t] * P_PG[t])
        # (2.17)
        # model.add_constraint(P_PG[t] <= W_set.L_Power[t])

    # 2.5 CCHP
    # 控制 CHP 开关
    u = {t: model.binary_var(name=f'u_{t}') for t in range(T)}
    # gas 输入量
    F_gas = {t: model.continuous_var(name=f'F_gas_{t}') for t in range(T)}
    # CHP 输出电量
    P_CHP = {t: model.continuous_var(name=f'P_CHP_{t}') for t in range(T)}
    # CHP 输出热量
    H_CHP = {t: model.continuous_var(name=f'H_CHP_{t}') for t in range(T)}
    # CHP 操作成本
    C_CHP = {t: model.continuous_var(name=f'C_CHP_{t}') for t in range(T)}
    for t in range(T):
        # (2.18)
        model.add_constraint(F_gas[t] == P_CHP[t] / (eta_chp_e * lambda_gas))
        # (2.19)
        model.add_constraint(H_CHP[t] == P_CHP[t] * eta_chp_h / eta_chp_e)
        # (2.20)
        model.add_constraint(u[t] * P_CHP_MIN <= P_CHP[t])
        model.add_constraint(u[t] * P_CHP_MAX >= P_CHP[t])
    for t in range(1, T):
        # (2.21)
        model.add_constraint(u[t] * P_RD_MIN <= P_CHP[t] - P_CHP[t - 1])
        model.add_constraint(u[t] * P_RU_MAX >= P_CHP[t] - P_CHP[t - 1])
    for t in range(T):
        # (2.22)
        model.add_constraint(C_CHP[t] == beta_chp * (P_CHP[t] + H_CHP[t]))

    # Cool
    # 电制冷机输入功率
    P_ec = {t: model.continuous_var(
        lb=P_EC_MIN, ub=P_EC_MAX, name=f'P_ec_{t}') for t in range(T)}
    # 吸收式制冷机输入功率
    H_ac = {t: model.continuous_var(
        lb=H_AC_MIN, ub=H_AC_MAX, name=f'H_ac_{t}') for t in range(T)}
    # 电制冷机成本
    C_ec = {t: model.continuous_var(name=f'C_ec_{t}') for t in range(T)}
    # 吸收式制冷机成本
    C_ac = {t: model.continuous_var(name=f'C_ac_{t}') for t in range(T)}
    for t in range(T):
        # (2.23)
        model.add_constraint(C_ec[t] == P_ec[t] * a_ec)
        # (2.24)
        # 见 P_ec 定义
        # (2.25)
        model.add_constraint(C_ac[t] == H_ac[t] * a_ac)
        # (2.26)
        # 见H_ac 定义

    # 2.7 Energy balance
    for t in range(T):
        # power balance
        model.add_constraint(
            P_PV[t] + P_PG[t] + P_BS_d[t] + P_CHP[t] == P_ec[t] + P_BS_c[t] + W_set.L_Power[t])
        # heat balance
        model.add_constraint(
            H_ac[t] + W_set.L_Heat[t] + H_TS_c[t] == H_TS_d[t] + H_CHP[t])
        # cool balance
        model.add_constraint(C_ec[t] + C_ac[t] == W_set.L_Cool[t])

    # 目标函数
    obj_expr = model.sum(C_PG[t] + C_CHP[t] + C_BS[t] for t in range(T))

    # 求解
    model.minimize(obj_expr)
    # solution: SolveSolution = model.solve(log_output=True)
    solution: SolveSolution = model.solve()
    # print(solution)
    variable_data = []
    if solution:
        print('objective value: ', solution.objective_value)
        variable_data.append(
            {"Variable": 'Obj_Value', "Value": solution.objective_value})
        for var in model.iter_variables():
            variable_data.append(
                {"Variable": var.name, "Value": solution[var]})
        if mode == 'w':
            df = pd.DataFrame(variable_data)
            df.to_excel(data_path, index=False)
            print('求解成功！')
            print(f'打开求解文件{data_path}查看结果')
    else:
        print(model.get_solve_details())
        print('solve_milp_all 求解失败！')
    return solution


def solve_milp_v(R_set: R, W_set: W, t, mode='r', data_path=''):
    """
    solve MILP model (known exogenous information)

    Args:
        R_set: R set
        W_set: W set
        mode: r or w
        data_path: ''

    Returns:
        solve details

    """
    # 创建模型
    model = Model(name='IETS MILP MODEL')

    # 决策变量
    E_TS = model.continuous_var(
        lb=E_TS_MIN, ub=E_TS_MAX, name=f'E_TS_t')
    E_BS = model.continuous_var(
        lb=E_BS_MIN, ub=E_BS_MAX, name=f'E_BS_t')

    # 创建约束
    # 2.1 BS
    # 电池放电量
    P_BS_d = model.continuous_var(
        lb=P_BS_D_MIN, ub=P_BS_D_MAX, name=f'P_BS_d_t')
    # 电池充电量
    P_BS_c = model.continuous_var(
        lb=P_BS_C_MIN, ub=P_BS_C_MAX, name=f'P_BS_c_t')
    # 电池放电状态 1/0
    a_BS_d = model.binary_var(name=f'a_BS_d_t')
    a_BS_c = model.binary_var(name=f'a_BS_c_t')
    # BS 成本
    C_BS = model.continuous_var(name=f'C_BS_t')
    # (2.1)
    model.add_constraint(P_BS_c <= a_BS_c * P_BS_C_MAX)
    # (2.2)
    model.add_constraint(P_BS_d <= a_BS_d * P_BS_D_MAX)
    # (2.3)
    model.add_constraint(a_BS_d + a_BS_c <= 1)
    # (2.4)
    # 略
    # (2.5)
    model.add_constraint(E_BS == (1 - eta_BS) * R_set.E_BS[t-1] +
                         eta_BS_c * P_BS_c - P_BS_d / eta_BS_d)
    # (2.6)
    # 见 E_BS 定义
    # (2.7)
    model.add_constraint(C_BS == beta_BS * (P_BS_c + P_BS_d))

    # 2.2 TS
    # TS 放热量
    H_TS_d = model.continuous_var(
        lb=H_TS_D_MIN, ub=H_TS_D_MAX, name=f'H_TS_d_t')
    # TS 储热量
    H_TS_c = model.continuous_var(
        lb=H_TS_C_MIN, ub=H_TS_C_MAX, name=f'H_TS_c_t')
    # 储热器放热状态 1/0
    a_TS_d = model.binary_var(name=f'a_TS_d_t')
    # 储热器蓄热状态 1/0
    a_TS_c = model.binary_var(name=f'a_TS_c_t')
    # (2.8)
    model.add_constraint(H_TS_c >= a_TS_c * H_TS_C_MIN)
    model.add_constraint(H_TS_c <= a_TS_c * H_TS_C_MAX)
    # (2.9)
    model.add_constraint(H_TS_d >= a_TS_d * H_TS_D_MIN)
    model.add_constraint(H_TS_d <= a_TS_d * H_TS_D_MAX)
    # (2.10)
    model.add_constraint(a_TS_c + a_TS_d <= 1)
    # (2.11)
    # 略
    # (2.12)
    model.add_constraint(E_TS == (
        1 - eta_TS) * R_set.E_TS[t - 1] + (eta_TS_c * H_TS_c - H_TS_d / eta_TS_d))
    # (2.13)
    # 见 E_TS 定义

    # 2.3 RES
    # 光伏输出电量
    P_PV = model.continuous_var(name=f'P_PV_t')
    # (2.14)
    # 略
    # (2.15)
    model.add_constraint(P_PV <= W_set.P_PV[t])

    # 2.4 PG
    # 从电网购电量
    P_PG = model.continuous_var(name=f'P_PG_t')
    # 购电成本
    C_PG = model.continuous_var(name=f'C_PG_t')
    # (2.16)
    model.add_constraint(C_PG == W_set.E_PRICE[t] * P_PG)
    # (2.17)
    # model.add_constraint(P_PG[t] <= W_set.L_Power[t])

    # 2.5 CCHP
    # 控制 CHP 开关
    u = model.binary_var(name=f'u_t')
    # gas 输入量
    F_gas = model.continuous_var(name=f'F_gas_t')
    # CHP 输出电量
    P_CHP = model.continuous_var(name=f'P_CHP_t')
    # CHP 输出热量
    H_CHP = model.continuous_var(name=f'H_CHP_t')
    # CHP 操作成本
    C_CHP = model.continuous_var(name=f'C_CHP_t')
    # (2.18)
    model.add_constraint(F_gas == P_CHP / (eta_chp_e * lambda_gas))
    # (2.19)
    model.add_constraint(H_CHP == P_CHP * eta_chp_h / eta_chp_e)
    # (2.20)
    model.add_constraint(u * P_CHP_MIN <= P_CHP)
    model.add_constraint(u * P_CHP_MAX >= P_CHP)
    # ! (2.21)
    # model.add_constraint(u * P_RD_MIN <= P_CHP - P_CHP[t - 1])
    # model.add_constraint(u * P_RU_MAX >= P_CHP - P_CHP[t - 1])
    # (2.22)
    model.add_constraint(C_CHP == beta_chp * (P_CHP + H_CHP))

    # Cool
    # 电制冷机输入功率
    P_ec = model.continuous_var(
        lb=P_EC_MIN, ub=P_EC_MAX, name=f'P_ec_t')
    # 吸收式制冷机输入功率
    H_ac = model.continuous_var(
        lb=H_AC_MIN, ub=H_AC_MAX, name=f'H_ac_t')
    # 电制冷机成本
    C_ec = model.continuous_var(name=f'C_ec_t')
    # 吸收式制冷机成本
    C_ac = model.continuous_var(name=f'C_ac_t')
    # (2.23)
    model.add_constraint(C_ec == P_ec * a_ec)
    # (2.24)
    # 见 P_ec 定义
    # (2.25)
    model.add_constraint(C_ac == H_ac * a_ac)
    # (2.26)
    # 见H_ac 定义

    # 2.7 Energy balance
    # power balance
    model.add_constraint(
        P_PV + P_PG + P_BS_d + P_CHP == P_ec + P_BS_c + W_set.L_Power[t])
    # heat balance
    model.add_constraint(
        H_ac + W_set.L_Heat[t] + H_TS_c == H_TS_d + H_CHP)
    # cool balance
    model.add_constraint(C_ec + C_ac == W_set.L_Cool[t])

    # 步长补充
    step_ts = 1
    step_bs = 1
    TS_SIZE = int(E_TS_MAX / step_ts)
    BS_SIZE = int(E_BS_MAX / step_bs)

    # 状态空间
    E_TS_SPACE = np.zeros((TS_SIZE, BS_SIZE))
    for j in range(BS_SIZE):
        E_TS_SPACE[:, j] = np.arange(0, TS_SIZE)

    E_BS_SPACE = np.zeros((TS_SIZE, BS_SIZE))
    for j in range(TS_SIZE):
        E_BS_SPACE[j, :] = np.arange(0, BS_SIZE)

    # 引入γ_g
    G = TS_SIZE * BS_SIZE
    gama_g = {g: model.binary_var(name=f'gema_{g}') for g in range(G)}
    model.add_constraint(model.sum(gama_g[g] for g in range(G)) == 1)

    # 引入 R_g
    model.add_constraint(E_TS == model.sum(
        gama_g[i * BS_SIZE + j] * E_TS_SPACE[i, j] for i in range(TS_SIZE) for j in
        range(BS_SIZE)))

    model.add_constraint(E_BS == model.sum(
        gama_g[i * BS_SIZE + j] * E_BS_SPACE[i, j] for i in range(TS_SIZE) for j in
        range(BS_SIZE)))

    # 修改 V
    V_t_plus_1 = 0
    gama_c = 0

    for E_TS in range(0, E_TS_MAX, 10):
        for E_BS in range(0, E_BS_MAX, 10):
            V_t_plus_1 += gama_g[gama_c] * v_table.get_value2((E_TS, E_BS), t)
            gama_c += 1
    # 目标函数
    obj_expr = C_PG + C_CHP + C_BS + V_t_plus_1

    # 求解
    model.minimize(obj_expr)
    # solution: SolveSolution = model.solve(log_output=True)
    solution: SolveSolution = model.solve()
    # print(solution)
    variable_data = []
    if solution:
        print('objective value: ', solution.objective_value)
        variable_data.append(
            {"Variable": 'Obj_Value', "Value": solution.objective_value})
        for var in model.iter_variables():
            variable_data.append(
                {"Variable": var.name, "Value": solution[var]})
        if mode == 'w':
            df = pd.DataFrame(variable_data)
            df.to_excel(data_path, index=False)
            print('求解成功！')
            print(f'打开求解文件{data_path}查看结果')
    else:
        print(model.get_solve_details())
        print('solve_milp_v 求解失败！')
    return solution


def solve_milp_mpc(R_t: R, W_t: W, t_start, t_end, n=3):
    """
    求解 MPC-n 的 milp
    参数：
        R_t: t_start 时刻的资源状态
        W_t: t_start-t_end 区间的环境信息
        n: MPC 预测步长
    返回：
        t_start 时刻的调度策略
    """
    # 创建模型
    model = Model(name='IETS MILP MODEL')

    # 决策变量
    E_TS = {t: model.continuous_var(
        lb=E_TS_MIN, ub=E_TS_MAX, name=f'E_TS_{t}') for t in range(t_start, t_end)}
    E_BS = {t: model.continuous_var(
        lb=E_BS_MIN, ub=E_BS_MAX, name=f'E_BS_{t}') for t in range(t_start, t_end)}
    # 初始资源状态
    model.add_constraint(E_BS[t_start] == R_t.E_BS[t_start])
    model.add_constraint(E_TS[t_start] == R_t.E_TS[t_start])
    # 创建约束
    # 2.1 BS
    # 电池放电量
    P_BS_D_MIN, P_BS_D_MAX = 0, 24  # 0-24kw

    P_BS_d = {t: model.continuous_var(lb=P_BS_D_MIN, ub=P_BS_D_MAX, name=f'P_BS_d_{t}') for t in
              range(t_start, t_end)}

    # 电池充电量
    P_BS_C_MIN, P_BS_C_MAX = 0, 24  # 0-24kw
    P_BS_c = {t: model.continuous_var(lb=P_BS_C_MIN, ub=P_BS_C_MAX, name=f'P_BS_c_{t}') for t in
              range(t_start, t_end)}
    # 电池放电状态 1/0
    a_BS_d = {t: model.binary_var(name=f'a_BS_d_{t}')
              for t in range(t_start, t_end)}
    a_BS_c = {t: model.binary_var(name=f'a_BS_c_{t}')
              for t in range(t_start, t_end)}
    # BS 成本
    C_BS = {t: model.continuous_var(name=f'C_BS_{t}')
            for t in range(t_start, t_end)}
    # BS 状态转移
    for t in range(t_start + 1, t_end):
        # (2.1)
        eta_BS_c, eta_BS_d = 0.98, 0.98
        model.add_constraint(E_BS[t] == E_BS[t - 1] +
                             eta_BS_c * P_BS_c[t] - P_BS_d[t] / eta_BS_d)
    for t in range(t_start, t_end):
        # (2.2)
        model.add_constraint(P_BS_c[t] <= a_BS_c[t] * P_BS_C_MAX)
        # (2.3)
        model.add_constraint(P_BS_d[t] <= a_BS_d[t] * P_BS_D_MAX)
        # (2.4)
        # 见 E_BS 定义
        # (2.5)
        model.add_constraint(a_BS_c[t] + a_BS_d[t] <= 1)
        # (2.6)
        beta_BS = 0.01
        model.add_constraint(C_BS[t] == beta_BS * (P_BS_c[t] + P_BS_d[t]))

    # 2.2 RES
    # (2.7)
    P_PV = {t: model.continuous_var(name=f'P_PV_{t}')
            for t in range(t_start, t_end)}

    for t in range(t_start, t_end):
        model.add_constraint(P_PV[t] <= W_t.P_PV[t])

    # 2.3 PG
    # (2.9)
    P_PG = {t: model.continuous_var(name=f'P_PG_{t}')
            for t in range(t_start, t_end)}  # t 时买入的电
    C_PG = {t: model.continuous_var(name=f'C_PG_{t}')
            for t in range(t_start, t_end)}
    for t in range(t_start, t_end):
        model.add_constraint(C_PG[t] == W_t.E_PRICE[t] * P_PG[t])

    # 2.4 CHP
    # (2.10)
    eta_chp_e = 0.3
    F_gas = {t: model.continuous_var(name=f'F_gas_{t}')
             for t in range(t_start, t_end)}
    P_CHP = {t: model.continuous_var(name=f'P_CHP_{t}')
             for t in range(t_start, t_end)}

    for t in range(t_start, t_end):
        model.add_constraint(F_gas[t] == P_CHP[t] / eta_chp_e)
    # (2.11)
    H_CHP = {t: model.continuous_var(name=f'H_CHP_{t}')
             for t in range(t_start, t_end)}
    eta_loss = 0.2
    eta_chp_h = 0.8
    for t in range(t_start, t_end):
        model.add_constraint(H_CHP[t] == P_CHP[t] *
                             eta_chp_h * (1 - eta_chp_e - eta_loss) / eta_chp_e)
    # (2.12)
    eta_gas = 3.24
    H_GAS = 9.78
    C_CHP = {t: model.continuous_var(name=f'C_CHP_{t}')
             for t in range(t_start, t_end)}
    for t in range(t_start, t_end):
        model.add_constraint(C_CHP[t] == eta_gas * F_gas[t] / H_GAS)

    # 2.5 TS
    eta_TS_d, eta_TS_c = 0.01, 0.98
    H_TS_D_MIN, H_TS_D_MAX = 0, 100
    H_TS_d = {t: model.continuous_var(lb=H_TS_D_MIN, ub=H_TS_D_MAX, name=f'H_TS_d_{t}') for t in
              range(t_start, t_end)}
    # 电池充电量
    H_TS_C_MIN, H_TS_C_MAX = 0, 100
    H_TS_c = {t: model.continuous_var(lb=H_TS_C_MIN, ub=H_TS_C_MAX, name=f'H_TS_c_{t}') for t in
              range(t_start, t_end)}
    # 储热器放热状态 1/0
    a_TS_d = {t: model.binary_var(name=f'a_TS_d_{t}')
              for t in range(t_start, t_end)}
    # 储热器蓄热状态 1/0
    a_TS_c = {t: model.binary_var(name=f'a_TS_c_{t}')
              for t in range(t_start, t_end)}
    for t in range(t_start + 1, t_end):
        # (2.13)
        model.add_constraint(E_TS[t] == (
            1 - eta_TS_d) * E_TS[t - 1] - H_TS_d[t] + eta_TS_c * H_TS_c[t])
    for t in range(t_start, t_end):
        # (2.14)~(2.18)
        model.add_constraint(H_TS_c[t] >= a_TS_c[t] * H_TS_C_MIN)
        model.add_constraint(H_TS_c[t] >= a_TS_c[t] * H_TS_C_MIN)
        model.add_constraint(H_TS_d[t] <= a_TS_d[t] * H_TS_D_MAX)
        model.add_constraint(H_TS_d[t] <= a_TS_d[t] * H_TS_D_MAX)
        model.add_constraint(a_TS_c[t] + a_TS_d[t] <= 1)
        # (2.18) 见 E_TS 定义

    # 2.6 Cool
    P_ec_MAX = 88.1
    H_ac_MAX = 168
    a_ec = 4
    a_ac = 0.7
    P_ec = {t: model.continuous_var(
        lb=0, ub=P_ec_MAX, name=f'P_ec_{t}') for t in range(t_start, t_end)}
    H_ac = {t: model.continuous_var(
        lb=0, ub=P_ec_MAX, name=f'H_ac_{t}') for t in range(t_start, t_end)}
    C_ec = {t: model.continuous_var(name=f'C_ec_{t}')
            for t in range(t_start, t_end)}
    C_ac = {t: model.continuous_var(name=f'C_ac_{t}')
            for t in range(t_start, t_end)}

    for t in range(t_start, t_end):
        # (2.19)
        model.add_constraint(C_ec[t] == P_ec[t] * a_ec)
        # (2.20)
        model.add_constraint(C_ac[t] == H_ac[t] * a_ac)
        # (2.21)~(2.22)
        # 见 P_ec 和 H_ac 定义

    # 2.7 Energy balance
    for t in range(t_start, t_end):
        # power balance
        model.add_constraint(
            P_PV[t] + P_PG[t] + P_BS_d[t] + P_CHP[t] == P_ec[t] + P_BS_c[t] + W_t.P_PG[t])
        # heat balance
        eta_he = 0.98
        model.add_constraint(H_ac[t] + W_t.H_TL[t] /
                             eta_he + H_TS_c[t] == H_TS_d[t] + H_CHP[t])
        # cool balance
        model.add_constraint(H_ac[t] * a_ac + P_ec[t] * a_ec == W_t.L_Cool[t])

    # 目标函数
    obj_expr = model.sum(C_PG[t] + C_CHP[t] + C_BS[t]
                         for t in range(t_start, t_end))

    # 求解
    model.minimize(obj_expr)
    # print('MILP 求解结果如下:')
    solution: SolveSolution = model.solve()
    variable_data = []

    if solution:
        # print(solution.get_value('a_BS_c_12'))
        # print(model.iter_variables())
        print('objective value: ', solution.objective_value)
        # print('decision variables: ')
        # for var in model.iter_variables():
        #     print(f"{var}: {solution[var]}")

        variable_data.append(
            {"Variable": 'Obj_Value', "Value": solution.objective_value})
        for var in model.iter_variables():
            variable_data.append(
                {"Variable": var.name, "Value": solution[var]})
        # print(model.get_solve_details())
        # if mode == 'w':
        #     df = pd.DataFrame(variable_data)
        #     df.to_excel(data_path, index=False)
        #     print('求解成功!')
        #     print(f'打开求解文件{data_path}查看结果')
    else:
        print(model.get_solve_details())
        print('求解失败！')
    return solution


if __name__ == '__main__':
    R_sets, W_sets = init_data(mode='r')
    for i in range(SAMPLE_SIZE):
        solve_milp_all(R_sets[i], W_sets[i], mode='w',
                       data_path=f'./data/Solution/Solution_{i}.xlsx')
