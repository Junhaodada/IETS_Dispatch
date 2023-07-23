from docplex.mp.model import Model
from MADP import X,V,S,W,R,C

def solve_milp(s_c:S, t):
    """
    solve (20a)
    """

    # OBJ
    # $X_t=argmin_{X_t \in \Pi _t} (C_t(S_t,X_t)+\sum_{g\in G}\gamma _g V_t^x(R_g))$

    s_c:S=S()

    # 创建模型
    model = Model(name='IETS MILP MODEL')

    # 定义决策变量
    T = 24  # 时间步数
    CHP_NUM = 1  # CHP设备数量
    RES_NUM = 1 # RES设备数量
    BS_NUM = 1 # BS设备数量
    TS_NUM = 1 # TS设备数量
    E_BUS_NUM = 6 # electricity buses
    T_NODE_NUM = 5 # thermal nodes
    H_CHP_MAX = 0
    H_CHP_MIN = 0
    P_CHP_MIN = 0
    P_CHP_MAX = 0
    H_TS_D_MAX = 0
    H_TS_D_MIN = 0
    H_TS_C_MAX = 0
    H_TS_C_MIN = 0
    P_BS_D_MIN = 0
    P_BS_D_MAX = 0
    P_BS_C_MIN = 0
    P_BS_C_MAX = 0
    P_RES_MIN = 0
    P_RES_MAX = 0
    Q_RES_MIN = 0
    Q_RES_MAX = 0
    H_CHP_n = {(n, t): model.continuous_var(lb=H_CHP_MIN, ub=H_CHP_MAX, name=f'H_CHP_{n}_{t}') for n in range(CHP_NUM) for t in range(T)}
    P_CHP_n = {(n, t): model.continuous_var(lb=P_CHP_MIN, ub=P_CHP_MAX, name=f'P_CHP_{n}_{t}') for n in range(CHP_NUM) for t in range(T)}
    a_BS_dn = {(n, t): model.binary_var(name=f'a_BS_d{n}_{t}') for n in range(BS_NUM) for t in range(T)}
    a_BS_cn = {(n, t): model.binary_var(name=f'a_BS_c{n}_{t}') for n in range(BS_NUM) for t in range(T)}
    a_TS_dn = {(n, t): model.binary_var(name=f'a_TS_d{n}_{t}') for n in range(TS_NUM) for t in range(T)}
    a_TS_cn = {(n, t): model.binary_var(name=f'a_TS_c{n}_{t}') for n in range(TS_NUM) for t in range(T)}
    H_TS_dn = {(n, t): model.continuous_var(lb=H_TS_D_MIN, ub=H_TS_D_MAX, name=f'H_TS_d{n}_{t}') for n in range(TS_NUM) for t in range(T)}
    H_TS_cn = {(n, t): model.continuous_var(lb=H_TS_C_MIN, ub=H_TS_C_MAX, name=f'H_TS_c{n}_{t}') for n in range(TS_NUM) for t in range(T)}
    P_BS_dn = {(n, t): model.continuous_var(lb=P_BS_D_MIN, ub=P_BS_D_MAX, name=f'P_BS_d{n}_{t}') for n in range(BS_NUM) for t in range(T)}
    P_BS_cn = {(n, t): model.continuous_var(lb=P_BS_C_MIN, ub=P_BS_C_MAX, name=f'P_BS_c{n}_{t}') for n in range(BS_NUM) for t in range(T)}
    P_RES_n = {(n, t): model.continuous_var(lb=P_RES_MIN, ub=P_RES_MAX, name=f'P_RES_{n}_{t}') for n in range(RES_NUM) for t in range(T)}
    Q_RES_n = {(n, t): model.continuous_var(lb=Q_RES_MIN, ub=Q_RES_MAX, name=f'Q_RES_{n}_{t}') for n in range(RES_NUM) for t in range(T)}
    gama_g = model.continuous_var_list()
    X_t = {}
    for t in range(T):
        X_t[t] = {
            'H_CHP_n': {n: H_CHP_n[n, t] for n in range(CHP_NUM)},
            'P_CHP_n': {n: P_CHP_n[n, t] for n in range(CHP_NUM)},
            'a_BS_dn': {n: a_BS_dn[n, t] for n in range(BS_NUM)},
            'a_BS_cn': {n: a_BS_cn[n, t] for n in range(BS_NUM)},
            'a_TS_dn': {n: a_TS_dn[n, t] for n in range(TS_NUM)},
            'a_TS_cn': {n: a_TS_cn[n, t] for n in range(TS_NUM)},
            'H_TS_dn': {n: H_TS_dn[n, t] for n in range(TS_NUM)},
            'H_TS_cn': {n: H_TS_cn[n, t] for n in range(TS_NUM)},
            'P_BS_dn': {n: P_BS_dn[n, t] for n in range(BS_NUM)},
            'P_BS_cn': {n: P_BS_cn[n, t] for n in range(BS_NUM)},
            'P_RES_n': {n: P_RES_n[n, t] for n in range(RES_NUM)},
            'Q_RES_n': {n: Q_RES_n[n, t] for n in range(RES_NUM)}
        }

    # 定义目标函数
    G = 0
    # obj_expr = [C(s_c,X_t[t])+ (model.sum(gama_g[g]) for g in range(G)) for t in range(T)]
    
    return X(t + 1)

