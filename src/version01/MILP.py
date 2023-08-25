from docplex.mp.model import Model
from params import *
import math


class W:
    """
    Exogenous information `W_t` of IETS includes stochastic processes of RES generation, 
    ambient temperature,electrical and thermal loads, and real-time price.
    """

    def __init__(self) -> None:
        self.E_PRICE = {t: 0 for t in range(T)}  # electrical real-time price
        self.P_EL = {(i, t): 0 for i in range(E_BUS_NUM)
                     for t in range(T)}  # active electrical load
        self.Q_EL = {(i, t): 0 for i in range(E_BUS_NUM)
                     for t in range(T)}  # reactive electrical load
        self.H_TL = {(n, t): 0 for n in range(TL_NUM)
                     for t in range(T)}  # thermal load in node
        self.TEM_AM = 0  # ambient temperature
        self.P_RES = {(n, t): 0 for n in range(RES_NUM)
                      for t in range(T)}  # Upper active power output limit of RES
        self.Q_RES = {(n, t): 0 for n in range(RES_NUM)
                      for t in range(T)}  # Upper reactive power output limit of RES


class R:
    """
    resource state `Rt` includes E_TS and E_BS (Energy stored in TS/BS at time-slot t)
    """

    def __init__(self) -> None:
        self.E_TS = {(n, t): 0 for n in range(TS_NUM)
                     for t in range(T)}  # 0-200 kWh
        self.E_BS = {(n, t): 0 for n in range(BS_NUM)
                     for t in range(T)}  # 0-80 kWh

    def set_E_TS(self, n, t, E_TS_var):
        self.E_TS = self.E_TS[(n, t)] = E_TS_var

    def get_E_TS(self, n, t):
        return self.E_TS[(n, t)]

    def set_E_BS(self, n, t, E_BS_var):
        self.E_BS = self.E_BS[(n, t)] = E_BS_var

    def get_E_BS(self, n, t):
        return self.E_BS[(n, t)]


class S:
    """
    System state `S_t={W_t, R_t}` includes exogeneous information 
    and resource state `R_t` where
    """

    def __init__(self, W: W, R: R) -> None:
        self.W = W
        self.R = R


def solve_milp(W_set: W, R_set: R, t):
    """
    solve (20a) MILP with cplex

    Parameter
    ---------
    W_set: W
        选取的W样本，包括24小时的W
    R_set: R
        选取的R样本，包括24小时的R
        举例：
            t时刻的第n个TS的E_TS表示为：
            >> R_set.E_TS[(n, t)]
            同理BS：
            >> R_set.E_BS[(n, t)]
    t: int
        t是算法1循环迭代的时刻
    """

    # 创建模型
    model = Model(name='IETS MILP MODEL')
    E_TS_n = {(n, t): model.continuous_var(lb=E_TS_MIN, ub=E_TS_MAX, name=f'E_TS_{n}_{t}') for n in range(TS_NUM) for t
              in range(T)}
    E_BS_n = {(n, t): model.continuous_var(lb=E_BS_MIN, ub=E_BS_MAX, name=f'E_BS_{n}_{t}') for n in range(BS_NUM) for t
              in range(T)}
    for n in range(TS_NUM):
        model.add_constraint(E_TS_n[(n, 0)] == R_set.E_TS[(n, 0)])
    for n in range(BS_NUM):
        model.add_constraint(E_BS_n[(n, 0)] == R_set.E_BS[(n, 0)])
    # 定义决策变量
    ########################################## 决策变量###############################################
    # CHP产热量
    H_CHP_N_MIN, H_CHP_N_MAX = 0, INF
    H_CHP_n = {(n, t): model.continuous_var(lb=H_CHP_N_MIN, ub=H_CHP_N_MAX,
                                            name=f'H_CHP_{n}_{t}') for n in range(CHP_NUM) for t in range(T)}
    # CHP产电量
    P_CHP_N_MIN, P_CHP_N_MAX = 0, INF
    P_CHP_n = {(n, t): model.continuous_var(lb=P_CHP_N_MIN, ub=P_CHP_N_MAX,
                                            name=f'P_CHP_{n}_{t}') for n in range(CHP_NUM) for t in range(T)}
    # 电池放电状态 1/0
    a_BS_dn = {(n, t): model.binary_var(name=f'a_BS_d_{n}_{t}')
               for n in range(BS_NUM) for t in range(T)}
    # 电池充电状态 1/0
    a_BS_cn = {(n, t): model.binary_var(name=f'a_BS_c_{n}_{t}')
               for n in range(BS_NUM) for t in range(T)}
    # 储热器放热状态 -1/0
    a_TS_dn = {(n, t): model.integer_var(lb=-1, ub=0, name=f'a_TS_d_{n}_{t}')
               for n in range(TS_NUM) for t in range(T)}
    # 储热器蓄热状态 1/0
    a_TS_cn = {(n, t): model.binary_var(name=f'a_TS_c_{n}_{t}')
               for n in range(TS_NUM) for t in range(T)}
    # 储热器状态 [-2,2]
    a_TS_n = {(n, t): model.integer_var(lb=-2, ub=2, name=f'a_TS_{n}_{t}')
              for n in range(TS_NUM) for t in range(T)}
    # a_EC_n = {(n, t): model.binary_var(name=f'a_EC_{n}_{t}')
    #           for n in range(EC_NUM) for t in range(T)}  # todo:cool var
    # a_AC_n = {(n, t): model.binary_var(name=f'a_AC_{n}_{t}')
    #           for n in range(AC_NUM) for t in range(T)}  # todo:cool var
    # 储热器放热能量
    H_TS_DN_MIN, H_TS_DN_MAX = 0, 100  # 0-100kw
    H_TS_dn = {(n, t): model.continuous_var(lb=H_TS_DN_MIN, ub=H_TS_DN_MAX,
                                            name=f'H_TS_d_{n}_{t}') for n in range(TS_NUM) for t in range(T)}
    # 储热器蓄热能量
    H_TS_CN_MIN, H_TS_CN_MAX = 0, 100  # 0-100kw
    H_TS_cn = {(n, t): model.continuous_var(lb=H_TS_CN_MIN, ub=H_TS_CN_MAX,
                                            name=f'H_TS_c_{n}_{t}') for n in range(TS_NUM) for t in range(T)}
    # 电池放电量
    P_BS_DN_MIN, P_BS_DN_MAX = 0, 24  # 0-24kw
    P_BS_dn = {(n, t): model.continuous_var(lb=P_BS_DN_MIN, ub=P_BS_DN_MAX,
                                            name=f'P_BS_d_{n}_{t}') for n in range(BS_NUM) for t in range(T)}
    # 电池充电量
    P_BS_CN_MIN, P_BS_CN_MAX = 0, 24  # 0-24kw
    P_BS_cn = {(n, t): model.continuous_var(lb=P_BS_CN_MIN, ub=P_BS_CN_MAX,
                                            name=f'P_BS_c_{n}_{t}') for n in range(BS_NUM) for t in range(T)}
    # 注释掉了 可再生能源实际产量 没有使用sy直接是输入可以不用，可再生能源的产量也可以通过温度计算产量（突出光伏）
    # P_RES_N_MIN, P_RES_N_MAX = 0, INF
    # P_RES_n = {(n, t): model.continuous_var(lb=P_RES_N_MIN, ub=P_RES_N_MAX,
    #                                         name=f'P_RES_{n}_{t}') for n in range(RES_NUM) for t in range(T)}
    # 可再生能源实际产量（同上）
    # Q_RES_N_MIN, Q_RES_N_MAX = 0, INF
    # Q_RES_n = {(n, t): model.continuous_var(lb=Q_RES_N_MIN, ub=Q_RES_N_MAX,
    #                                         name=f'Q_RES_{n}_{t}') for n in range(RES_NUM) for t in range(T)}

    #####################################模型约束#######################################################################
    ####################################(1)TS约束#######################################################################
    # 补充的决策变量
    m_TS_n = {(n, t): 0.8 * model.binary_var(name=f'm_TS_{n}_{t}') for n in range(TS_NUM) for t in range(T)}
    tem_TS_N_MIN, tem_TS_N_MAX = 25, 100  # 已修改 下界修改成20，30以上可以输出，20-30说明在充热量，管道温度的下界是20
    tem_TS_n = {(n, t): model.continuous_var(lb=tem_TS_N_MIN, ub=tem_TS_N_MAX,
                                             name=f'tem_TS_{n}_{t}') for n in range(TS_NUM) for t in range(T)}
    tem_TS_S_n = {(n, t): model.continuous_var(lb=tem_TS_N_MIN, ub=tem_TS_N_MAX,
                                               name=f'tem_TS_S_{n}_{t}') for n in range(TS_NUM) for t in range(T)}
    tem_TS_R_n = {(n, t): model.continuous_var(lb=tem_TS_N_MIN, ub=tem_TS_N_MAX,
                                               name=f'tem_TS_R_{n}_{t}') for n in range(TS_NUM) for t in range(T)}
    H_TS_N_MIN, H_TS_N_MAX = 0, 100
    H_TS_n = {(n, t): model.continuous_var(lb=H_TS_N_MIN, ub=H_TS_N_MAX,
                                           name=f'H_TS_{n}_{t}') for n in range(TS_NUM) for t in range(T)}

    # 补充的约束
    # 1a
    for n in range(TS_NUM):
        for t in range(T):
            model.add_constraint(m_TS_n[(n, t)] == 0.8 * a_TS_cn[(n, t)] - 0.8 * a_TS_dn[(n, t)])

    # 1b
    for n in range(TS_NUM):
        for t in range(T):
            model.add_constraint(H_TS_cn[(n, t)] >= a_TS_cn[(n, t)] * H_TS_CN_MIN)
            model.add_constraint(H_TS_cn[(n, t)] <= a_TS_cn[(n, t)] * H_TS_CN_MAX)

    # 1c
    for n in range(TS_NUM):
        for t in range(T):
            model.add_constraint(H_TS_dn[(n, t)] >= a_TS_dn[(n, t)] * H_TS_DN_MIN)
            model.add_constraint(H_TS_dn[(n, t)] <= a_TS_dn[(n, t)] * H_TS_DN_MIN)

    # 1d
    for n in range(TS_NUM):
        for t in range(T):
            model.add_constraint(a_TS_cn[(n, t)] + a_TS_dn[(n, t)] <= 1)

    # 1e
    for n in range(TS_NUM):
        for t in range(T):
            model.add_constraint(a_TS_n[(n, t)] == a_TS_dn[(n, t)] + a_TS_cn[(n, t)])

    # 1f
    for n in range(TS_NUM):
        for t in range(T - 1):
            model.add_constraint(E_TS_n[(n, t + 1)] == (
                    1 - 0.01) * E_TS_n[(n, t)] - H_TS_dn[(n, t)] + 0.98 * H_TS_cn[(n, t)])

    # 1g
    C_wt, p_water, V_TS_n, tem_TS = 4.182 / 3600, 1000, 2.46, 30
    for n in range(TS_NUM):
        for t in range(T):
            model.add_constraint(
                E_TS_n[(n, t)] == C_wt * p_water * V_TS_n * (tem_TS_n[(n, t)] - tem_TS))

    # 1h 已修改  不是决策变量报错了#sy修改成决策变量，初始时间作为约束，随之确定了初始时间TS的温度（公式1g中唯一的变量）
    # 见定义
    # 1i
    # 变量已经初始化

    # 1j
    for n in range(TS_NUM):
        for t in range(T):
            model.add_constraint(
                tem_TS_S_n[(n, t)] >= a_TS_cn[(n, t)] * tem_TS_n[(n, t)])
            model.add_constraint(
                tem_TS_R_n[(n, t)] >= a_TS_cn[(n, t)] * tem_TS_n[(n, t)])
            model.add_constraint(tem_TS_S_n[(n, t)] <= a_TS_dn[(
                n, t)] * tem_TS_n[(n, t)] + (1 - a_TS_dn[(n, t)]) * tem_TS_N_MAX)
            model.add_constraint(tem_TS_R_n[(n, t)] <= a_TS_dn[(
                n, t)] * tem_TS_n[(n, t)] + (1 - a_TS_dn[(n, t)]) * tem_TS_N_MAX)

    # 1k
    for n in range(TS_NUM):
        for t in range(T):
            model.add_constraint(H_TS_dn[(n, t)] == a_TS_dn[(
                n, t)] * 0.8 * C_wt * (tem_TS_S_n[(n, t)] - tem_TS_R_n[(n, t)]))

    # 1l
    for n in range(TS_NUM):
        for t in range(T):
            model.add_constraint(H_TS_cn[(n, t)] == a_TS_cn[(
                n, t)] * 0.8 * C_wt * (tem_TS_S_n[(n, t)] - tem_TS_R_n[(n, t)]))

    # 1m
    for n in range(TS_NUM):
        for t in range(T):
            model.add_constraint(
                H_TS_n[(n, t)] == H_TS_cn[(n, t)] - H_TS_dn[(n, t)])

    # 1n
    for n in range(TS_NUM):
        for t in range(T):
            model.add_constraint(tem_TS_S_n[(n, t)] >= 20)
            model.add_constraint(tem_TS_S_n[(n, t)] <= 100)
            model.add_constraint(tem_TS_R_n[(n, t)] >= 20)
            model.add_constraint(tem_TS_R_n[(n, t)] <= 100)

    # ! 1o
    for n in range(TS_NUM):
        model.add_constraint(E_TS_n[(n, 0)] == E_TS_n[(n, T - 1)])

    ##########################################(2)CHP约束##################################################################
    # 补充的决策变量
    CHP_P_NUM = 4
    sigma_CHP_n_c = {(c, n, t): model.continuous_var(lb=0, name=f'sigma_CHP_{c}_{n}_{t}')
                     for c in range(CHP_P_NUM) for n in range(CHP_NUM) for t in range(T)}
    H_CHP_n_c = [0, 300, 300, 0]
    P_CHP_n_c = [0, 120, 180, 300]
    tem_CHP_MIN, tem_CHP_MAX = 20, 100
    tem_CHP_n = {(n, t): model.continuous_var(lb=tem_CHP_MIN, ub=tem_CHP_MAX,
                                              name=f'tem_CHP_{n}_{t}') for n in range(CHP_NUM) for t in range(T)}
    tem_CHP_S_n = {(n, t): model.continuous_var(lb=tem_CHP_MIN, ub=tem_CHP_MAX,
                                                name=f'tem_CHP_S_{n}_{t}') for n in range(CHP_NUM) for t in range(T)}
    tem_CHP_R_n = {(n, t): model.continuous_var(lb=tem_CHP_MIN, ub=tem_CHP_MAX,
                                                name=f'tem_CHP_R_{n}_{t}') for n in range(CHP_NUM) for t in range(T)}
    m_CHP_n = 3.9
    # 补充的约束
    # 2a
    for c in range(CHP_P_NUM):
        for n in range(CHP_NUM):
            for t in range(T):
                model.add_constraint(H_CHP_n[(n, t)] == model.sum(sigma_CHP_n_c[(
                    c, n, t)] * H_CHP_n_c[c]))
    # 2b
    for c in range(CHP_P_NUM):
        for n in range(CHP_NUM):
            for t in range(T):
                model.add_constraint(P_CHP_n[(n, t)] == model.sum(sigma_CHP_n_c[(c,
                                                                                 n, t)] * P_CHP_n_c[c]))

    # 2c
    for c in range(CHP_P_NUM):
        for n in range(CHP_NUM):
            for t in range(T):
                model.add_constraint(model.sum(sigma_CHP_n_c[(c, n, t)]) == 1)
    # 2d
    # 见sigma_CHP_n_c定义
    # 2e
    # 见Q_CHP_n定义
    # 2f
    for n in range(CHP_NUM):
        for t in range(T):
            model.add_constraint(H_CHP_n[(n, t)] == C_wt * m_CHP_n * (
                    tem_CHP_S_n[(n, t)] - tem_CHP_R_n[(n, t)]))
    # 2g
    # 见tem_CHP_n定义
    # 2h
    a_n = [0, 0.018, 0.015, 0.00024, 0.0013, 0.0013]
    C_CHP_n_MIN, C_CHP_n_MAX = 0, INF
    C_CHP_n = {(n, t): model.continuous_var(lb=C_CHP_n_MIN, ub=C_CHP_n_MAX,
                                            name=f'C_CHP_n{n}_{t}') for n in range(CHP_NUM) for t in range(T)}
    for n in range(CHP_NUM):
        for t in range(T):
            model.add_constraint(
                C_CHP_n[(n, t)] == a_n[5] * P_CHP_n[(n, t)] ** 2 + a_n[4] * H_CHP_n[(n, t)] ** 2 + a_n[3] * P_CHP_n[(
                    n, t)] * H_CHP_n[(n, t)] + a_n[2] * P_CHP_n[(n, t)] + a_n[1] * H_CHP_n[(n, t)] + a_n[0])
    ###########################################(3)热负载约束###############################################################
    # 补充的决策变量
    H_TL_MIN, H_TL_MAX = 0, INF
    H_TL_n = {(n, t): model.continuous_var(lb=H_TL_MIN, ub=H_TL_MAX,
                                           name=f'H_TL_{n}_{t}') for n in range(TL_NUM) for t in range(T)}
    tem_TL_S_n_MIN, tem_TL_S_n_MAX = 20, 100
    tem_TL_R_n_MIN, tem_TL_R_n_MAX = 20, 70
    tem_TL_S_n = {(n, t): model.continuous_var(lb=tem_TL_S_n_MIN, ub=tem_TL_S_n_MAX,
                                               name=f'tem_TL_S_{n}_{t}') for n in range(TL_NUM) for t in range(T)}
    tem_TL_R_n = {(n, t): model.continuous_var(lb=tem_TL_R_n_MIN, ub=tem_TL_R_n_MAX,
                                               name=f'tem_TL_R_{n}_{t}') for n in range(TL_NUM) for t in range(T)}
    # m_TL_MIN, m_TL_MAX = 0, 100
    # m_TL_n = {(n, t): model.continuous_var(lb=m_TL_MIN, ub=m_TL_MAX,
    #                                        name=f'm_TL_{n}_{t}') for n in range(TL_NUM) for t in range(T)}
    m_TL_n = TL_NUM
    # 3a
    for n in range(TL_NUM):
        for t in range(T):
            model.add_constraint(H_TL_n[(n, t)] == C_wt * m_TL_n * (
                    tem_TL_S_n[(n, t)] - tem_TL_R_n[(n, t)]))
    # 3b
    # 见tem_TL_S_n, tem_TL_R_n定义
    ########################################### (4)TDN/热配网约束###########################################################
    # input your code
    # 参数
    nodes_in_k_NUM = 1  # 进入节点k的顶点集合
    nodes_out_k_NUM = 1  # 出节点k的顶点集合
    tem_AM = {t: 30 for t in range(T)}  # 环境温度已给的一组数据
    # L_kl[(k, l)]   # 管道长度，输入数据
    # mu_pipe_kl[(k, l)]  # 单位长度管道总传热系数，因此需要与L_kl相乘，输入数据

    # 决策变量10
    # tem_pipe_S_out_kl  # sy供应管道从k流到l的温度,kl的出口温度，对l求和就是从k分流出去的温度
    # tem_pipe_S_in_kl  # sy供应管道流入kl的温度，kl的进口温度，即为顶点k处的温度
    # tem_pipe_R_out_kl
    # tem_pipe_R_in_kl
    # tem_node_S_k  # 在供应管道中，节点k的热流温度
    # tem_node_R_k  # 在回流管道中，节点k的热流温度
    # m_pipe_S_kl  # 供应管道kl的质量流
    # m_pipe_R_kl  # 回流管道kl的质量流
    # d_pipe_S_kl  # 管道温度下降系数，为什么是变量呢？因为依赖于管道中质量流，见4m
    # d_pipe_R_kl  # 管道温度下降系数，为什么是变量呢？因为依赖于管道中质量流, 见4m
    # H_CHP_n_c  # 热产量，需要解决一个问题是极值点如何生成
    # P_CHP_n_c  # 电产量，需要解决一个问题是极值点如何生成
    tem_pipe_S_out_kl = {(l, t): model.continuous_var(
        lb=0, ub=100, name=f'tem_pipe_S_out_{l}_{t}') for l in range(nodes_out_k_NUM) for t in range(T)}
    tem_pipe_S_in_kl = {(l, t): model.continuous_var(
        lb=0, ub=100, name=f'tem_pipe_S_in_{l}_{t}') for l in range(nodes_in_k_NUM) for t in range(T)}
    tem_pipe_R_out_kl = {(l, t): model.continuous_var(
        lb=0, ub=100, name=f'tem_pipe_R_out_{l}_{t}') for l in range(nodes_out_k_NUM) for t in range(T)}
    tem_pipe_R_in_kl = {(l, t): model.continuous_var(
        lb=0, ub=100, name=f'tem_pipe_R_in_{l}_{t}') for l in range(nodes_in_k_NUM) for t in range(T)}
    tem_node_S_k = {(k, t): model.continuous_var(
        lb=0, ub=100, name=f'tem_node_S_{k}_{t}') for k in range(TL_NUM) for t in range(T)}
    tem_node_R_k = {(k, t): model.continuous_var(
        lb=0, ub=100, name=f'tem_node_R_{k}_{t}') for k in range(TL_NUM) for t in range(T)}
    m_pipe_S_kl = {(k, l, t): model.continuous_var(lb=0, ub=100, name=f'm_pipe_S_{k}_{l}_{t}')
                   for k in range(TL_NUM) for l in range(nodes_out_k_NUM) for t in range(T)}
    m_pipe_R_kl = {(k, l, t): model.continuous_var(lb=0, ub=100, name=f'm_pipe_R_{k}_{l}_{t}')
                   for k in range(TL_NUM) for l in range(nodes_out_k_NUM) for t in range(T)}
    d_pipe_S_kl = {(k, l, t): model.continuous_var(lb=0, ub=100, name=f'd_pipe_Skl_{k}_{l}_{t}')
                   for k in range(TL_NUM) for l in range(nodes_out_k_NUM) for t in range(T)}
    d_pipe_R_kl = {(k, l, t): model.continuous_var(lb=0, ub=100, name=f'd_pipe_Rkl_{k}_{l}_{t}')
                   for k in range(TL_NUM) for l in range(nodes_out_k_NUM) for t in range(T)}

    # 补充的约束a_TS_dn': {n: a_TS_dn[n, t] for n in range(TS_NUM)},

    # 4g
    m_TL_k = 3.9  # 已修改 ?constant=3.9，需要修改  错误需要修改
    m_CHP_k_MIN, m_CHP_k_MAX = 0, INF  # ?
    m_CHP_k = {(k, t): model.continuous_var(
        lb=m_CHP_k_MIN, ub=m_CHP_k_MAX, name=f'm_CHP_{k}_{t}') for k in range(TL_NUM) for t in range(T)}
    # 已修改 ?同上-1/0.8，根据TS运行状态m_TS_k=0或0.8，不是连续变量是固定的
    m_TS_k = {(k, t): -1.8 * model.binary_var(name=f'm_TS_{k}_{t}') for k in range(TL_NUM) for t in range(T)}
    m_pipe_S_jk_MIN, m_pipe_S_jk_MAX = -1, 3.9  # 已修改 ?sy自己设定的，文章给了每个节点质量流但是不知道怎么用，因为质量流质变的0-3.9
    m_pipe_S_jk = {(j, k, t): model.continuous_var(lb=m_pipe_S_jk_MIN, ub=m_pipe_S_jk_MAX, name=f'm_pipe_S_{j}_{k}_{t}')
                   for j in range(nodes_in_k_NUM) for k in range(TL_NUM) for t in range(T)}
    for l in range(nodes_out_k_NUM):
        for j in range(nodes_in_k_NUM):
            for k in range(TL_NUM):
                for t in range(T):
                    model.add_constraint(
                        model.sum(m_pipe_S_kl[(k, l, t)]) - model.sum(m_pipe_S_jk[(j, k, t)]) == m_CHP_k[(k, t)] -
                        m_TL_k - m_TS_k[(k, t)])
    # 4h
    m_pipe_R_jk_MIN, m_pipe_R_jk_MAX = -1, 3.9  #  已修改 ?sy同上
    m_pipe_R_jk = {(j, k, t): model.continuous_var(lb=m_pipe_R_jk_MIN, ub=m_pipe_R_jk_MAX, name=f'm_pipe_S_{j}_{k}_{t}')
                   for j in range(nodes_in_k_NUM) for k in range(TL_NUM) for t in range(T)}
    for l in range(nodes_in_k_NUM):
        for j in range(nodes_out_k_NUM):
            for k in range(TL_NUM):
                for t in range(T):
                    model.add_constraint(
                        model.sum(m_pipe_R_kl[(k, l, t)]) - model.sum(m_pipe_R_jk[(j, k, t)]) == m_TL_k -
                        m_CHP_k[(k, t)] + m_TS_k[(k,t)])
    # 4a
    a_TS_dk = {(k, t): -1*model.binary_var(name=f'a_TS_dk_{k}_{t}') for k in range(TS_NUM) for t in range(T)}
    tem_TS_S_k_MIN, tem_TS_S_k_MAX = 0, INF  # ?
    tem_TS_S_k = {(k, t): model.continuous_var(
        lb=tem_TS_S_k_MIN, ub=tem_TS_S_k_MAX, name=f'tem_TS_Sk_{k}_{t}') for k in range(TL_NUM) for t in range(T)}
    tem_TS_R_k_MIN, tem_TS_R_k_MAX = 0, INF  # ?
    tem_TS_R_k = {(k, t): model.continuous_var(
        lb=tem_TS_R_k_MIN, ub=tem_TS_R_k_MAX, name=f'tem_TS_Rk_{k}_{t}') for k in range(TL_NUM) for t in range(T)}
    for l in range(nodes_in_k_NUM):
        for k in range(TL_NUM):
            for t in range(T):
                model.add_constraint(tem_node_S_k[(k, t)] * (
                        model.sum(m_pipe_S_kl[(k, l, t)]) + m_CHP_k[(k, t)] + a_TS_dk[(k, t)] * 0.8) ==
                                     model.sum(tem_pipe_S_out_kl[(l, t)] * m_pipe_S_kl[(k, l, t)]) + tem_CHP_S_n[
                                         (k, t)] * m_CHP_k[(k, t)] + a_TS_dk[(k, t)] * 0.8 * tem_TS_S_k[(k, t)])
    # 4b
    a_TS_ck_MIN, a_TS_ck_MAX = -1, 1  # 已修改 ?sy说明charging:1/0;discharging:-1/0.约束4a需要引入变量a_TS_n[(n,t)]=a_TS_cn[(n,t)]+a_TS_dn[(n,t)]
    a_TS_ck = {(k, t): model.integer_var(
        lb=a_TS_ck_MIN, ub=a_TS_ck_MAX, name=f'a_TS_c_{k}_{t}') for k in range(TL_NUM) for t in range(T)}
    m_TL_k = 3.9  # 已修改 ?constant=3.9，需要修改  错误需要修改

    for k in range(TL_NUM):
        for t in range(T):
            for l in range(nodes_out_k_NUM):
                model.add_constraint(tem_node_R_k[(k, t)] * (
                        model.sum(m_pipe_S_kl[(k, l, t)]) + m_TL_k + a_TS_ck[(k, t)] * 0.8) == model.sum(
                    tem_pipe_R_out_kl[(l, t)] * m_pipe_R_kl[(
                        k, l, t)] + tem_TL_R_n[(k, t)] * m_CHP_k[(k, t)] + a_TS_ck[(k, t)] * 0.8 * tem_TS_R_k[(k, t)]))
    # 4c
    tem_TL_S_k_MIN, tem_TL_S_k_MAX = 20, 100  # 已修改 ?sy供应管道温度上下界20-100，回流管道20-70
    tem_TL_S_k = {(k, t): model.continuous_var(
        lb=tem_TL_S_k_MIN, ub=tem_TL_S_k_MAX, name=f'tem_TL_S_{k}_{t}') for k in range(TL_NUM) for t in range(T)}

    for l in range(nodes_out_k_NUM):
        for k in range(TL_NUM):
            for t in range(T):
                model.add_constraint(tem_node_S_k[(k, t)] == tem_pipe_S_in_kl[(l, t)] == tem_TL_S_k[(k, t)])
    # 4d
    for k in range(TL_NUM):
        for t in range(T):
            model.add_constraint(tem_node_S_k[(k, t)] + (1 - a_TS_ck[(k, t)]) * (tem_TS_N_MIN - tem_node_S_k[(
                k, t)]) <= tem_TS_S_k[(k, t)])
            model.add_constraint(tem_node_S_k[(k, t)] + (1 - a_TS_ck[(k, t)]) * (tem_TS_N_MAX - tem_node_S_k[(
                k, t)]) >= tem_TS_S_k[(k, t)])
    # 4e
    tem_CHP_R_k_MIN, tem_CHP_R_k_MAX = 20, 100  # 已修改 ?sy20-100
    tem_CHP_R_k = {(k, t): model.continuous_var(
        lb=tem_CHP_R_k_MIN, ub=tem_CHP_R_k_MAX, name=f'tem_CHP_R_{k}_{t}') for k in range(TL_NUM) for t in range(T)}
    for l in range(nodes_out_k_NUM):
        for k in range(TL_NUM):
            for t in range(T):
                model.add_constraint(tem_node_R_k[(k, t)] == tem_pipe_R_in_kl[(l, t)] == tem_CHP_R_k[(
                    k, t)])
    # 4f
    for k in range(TL_NUM):
        for t in range(T):
            model.add_constraint(tem_node_R_k[(k, t)] + (1 - a_TS_dk[(k, t)]) * (tem_TS_N_MIN - tem_node_R_k[(
                k, t)]) <= tem_TS_R_k[(k, t)])
            model.add_constraint(tem_node_R_k[(k, t)] + (1 - a_TS_dk[(k, t)]) * (tem_TS_N_MAX - tem_node_R_k[(
                k, t)]) >= tem_TS_R_k[(k, t)] for k in range(TL_NUM) for t in range(T))
    # 4l 由热损失引起的管道从入口到出口的流量温度下降。
    Line_NUM = 6  # ?BUS是6条（小规模的）第6条电力母线和5个热节点，
    for k in range(TL_NUM):
        for l in range(nodes_in_k_NUM):
            for t in range(T):
                model.add_constraint(tem_pipe_R_out_kl[(k, l, t)] == d_pipe_R_kl[(
                    k, l, t)] * (tem_pipe_R_in_kl[(k, l, t)] - tem_AM[t]) + tem_AM[t])
    for k in range(TL_NUM):
        for l in range(nodes_in_k_NUM):
            for t in range(T):
                model.add_constraint(tem_pipe_S_out_kl[(k, l, t)] == d_pipe_S_kl[(
                    k, l, t)] * (tem_pipe_S_in_kl[(k, l, t)] - tem_AM[t]) + tem_AM[t])
    # 由于温度和质量流都是变量，会出现4a是非凸的，但通过引入二进制变量变呈线性的，也就是保证质量流可以确定地求出来。质量流变量的原因：1、至少有一个CHP的质量流是变化的，为了满足负载需求；2、由于储热器充放热状态。
    # 由于以上两点导致所有节点质量流是变量，其中一个节点是CHP节点。我们考虑的中型模型有两个TS。我们就按照两个TS，引入二进制变量。保证4a线性化，上面的约束保证了质量流是唯一的。
    # 引入向量phi=(a_TS_n[(n,t)])关于每一维表示第几个TS对应的状态，需增加4i-4k约束4m-4n，由于我不知道怎么表示向量，师弟补充一下这几个约束。
    # phi[t]=(1,0,...),t时刻的phi
    phi_NUM = 3 ** TS_NUM  # 已修改 这里小型是1个TS，中型2个TS，先考虑前一种情况。此处设置需修改
    phi = {(t, n): (a_TS_n[(n, t)]) for n in range(TS_NUM) for t in range(T)}
    # 4i
    m_pipe_S_kl_phi = {(k, l, p, t): model.continuous_var(lb=0, ub=100, name=f'm_pipe_S_{k}_{l}_{p}_{t}')
                       for p in range(phi_NUM) for k in range(TL_NUM) for l in range(nodes_out_k_NUM) for t in
                       range(T)}
    for k in range(TL_NUM):
        for l in range(nodes_out_k_NUM):
            for t in range(T):
                for p in range(phi_NUM):
                    model.add_constraint(m_pipe_R_kl[(k, l, t)] == model.sum(p * m_pipe_S_kl_phi[(k, l, p, t)]))
    # 4j 质量流一般比较小3.9或者0.8（文中TS是0.8，CHP是3.9），设置的上下界0-100也可以，如果实验出问题可以再修改.
    m_CHP_k_phi = {(k, p, t): model.continuous_var(lb=0, ub=100, name=f'm_pipe_S_{k}_{l}_{p}_{t}')
                   for p in range(phi_NUM) for k in range(TL_NUM) for l in range(nodes_out_k_NUM) for t in range(T)}
    for p in range(phi_NUM):
        for k in range(TL_NUM):
            for t in range(T):
                model.add_constraint(m_CHP_k == model.sum(p * m_CHP_k_phi[(k, p, t)]))

    # 4k phi不是约束#sy是决策变量，在多个TS的情况下才会有用，但是由于引入phi表示这么多个TS只能有一个在运行，因为要求求和为1
    for t in range(T):
        for n in range(TS_NUM):
            model.add_constraint(model.sum(phi[t][n]) == 1)

    # 4m C_wt=4.182/3600，错误需要修改
    mu_pipe_kl_MIN, mu_pipe_kl_MAX = 0, INF  # ?sy单位长度管道总传热系数，因此需要与L_kl相乘，输入数据已经给出了
    mu_pipe_kl = {(k, l, t): model.continuous_var(lb=mu_pipe_kl_MIN, ub=mu_pipe_kl_MAX, name=f'mu_pipe_kl_{k}_{l}_{t}')
                  for k in range(TL_NUM) for l in range(nodes_out_k_NUM) for t in range(T)}
    L_kl_MIN, L_kl_MAX = 0, INF  # ?
    L_kl = {(k, l): model.continuous_var(lb=L_kl_MIN, ub=L_kl_MAX, name=f'L_kl_{k}_{l}')
            for k in range(TL_NUM) for l in range(nodes_out_k_NUM)}
    for k in range(TL_NUM):
        for l in range(nodes_in_k_NUM):
            for t in range(T):
                model.add_constraint(d_pipe_R_kl[(k, l, t)] == math.exp(
                    -(mu_pipe_kl[(k, l, t)] * L_kl[(k, l)]) / (C_wt * m_pipe_R_kl[(k, l, t)])))

    # for k in range(TL_NUM):
    #     for l in range(nodes_in_k_NUM):
    #         for t in range(T):
    #             model.add_constraint(d_pipe_S_kl[(k, l, t)] == math.exp(-(mu_pipe_kl[(k, l, t)] * L_kl[(
    #     k, l)]) / (C_wt * m_pipe_S_kl[(k, l, t)])) for kl in range(Line_NUM) for t in range(T))
    # end input
    ############################################(5)RES约束✔#################################################################
    # 5a
    # 见P_RES_n定义
    # 5b
    # delete
    ###########################################(6)BS约束✔####################################################################
    # 补充的决策变量
    C_BS_MIN, C_BS_MAX = 0, INF
    C_BS_n = {(n, t): model.continuous_var(lb=C_BS_MIN, ub=C_BS_MAX,
                                           name=f'C_BS_{n}_{t}') for n in range(BS_NUM) for t in range(T)}
    # 6a
    for n in range(BS_NUM):
        for t in range(T):
            model.add_constraint(
                P_BS_cn[(n, t)] <= a_BS_cn[(n, t)] * P_BS_CN_MAX)

    # 6b
    for n in range(BS_NUM):
        for t in range(T):
            model.add_constraint(
                P_BS_dn[(n, t)] <= a_BS_dn[(n, t)] * P_BS_DN_MAX)

    # 6c不是决策
    # 见定义

    # 6d
    for n in range(BS_NUM):
        for t in range(T - 1):  # Note: Starting from t=1 to avoid t-1 when t=0
            model.add_constraint(E_BS_n[(n, t + 1)] == E_BS_n[(
                n, t)] + 0.98 * P_BS_cn[(n, t + 1)] - P_BS_dn[(n, t + 1)] / 0.98)

    # 6e
    for n in range(BS_NUM):
        model.add_constraint(E_BS_n[(n, T - 1)] == 0.5 * E_BS_MAX)
        model.add_constraint(E_BS_n[(n, T)] == E_BS_n[(n, T - 1)])

    # 6f
    for n in range(BS_NUM):
        for t in range(T):
            model.add_constraint(a_BS_dn[(n, t)] + a_BS_cn[(n, t)] <= 1)

    # 6g
    for n in range(BS_NUM):
        for t in range(T):
            model.add_constraint(
                C_BS_n[(n, t)] == 0.01 * (P_BS_cn[(n, t)] + P_BS_dn[(n, t)]))

    ########################################## (7)EDN/电配网约束##############################################################
    # input your code
    # 参数

    # E_BUS_NUM = 1  # 母线的数量不应该是数量，应该是集合
    Line_NUM = 10  # 母线构成的集合，怎么表示？
    P_EL_i = 10  # 节点i处的负载，sy根据点负载，以及Active power ratio、Reactive power ratio，可以知道每个节点需求量
    P_BUS_ij_MAX = 10  # ? 上界，
    u_BUS_MAX = 10  # 电压上限sy数据不对
    u_BUS_MIN = 10  # 电压下限
    # 决策变量
    r_ij = {(i, j): 0 for i in range(E_BUS_NUM) for j in range(E_BUS_NUM)}  # ?母线ij电阻已知，不应该属于决策变量，sy对的数据已知
    x_ij = {(i, j): 0 for i in range(E_BUS_NUM) for j in range(E_BUS_NUM)}  # ?母线ij电抗已知，不应该属于决策变量，sy对的数据已知
    P_i = {(i, t): 0 for i in range(E_BUS_NUM) for t in range(T)}  # ?注入到节点i的有功功率sy√
    Q_i = {(i, t): 0 for i in range(E_BUS_NUM) for t in range(T)}  # ?注入到节点i的无功功率sy√
    P_ij = {(i, j, t): 0 for i in range(E_BUS_NUM) for j in range(E_BUS_NUM) for t in range(T)}  # 母线ij的有功功率
    Q_ij = {(i, j, t): 0 for i in range(E_BUS_NUM) for j in range(E_BUS_NUM) for t in range(T)}  # 母线ij的无功功率
    u_i = {(i, t): 0 for i in range(E_BUS_NUM) for t in range(T)}  # ?DEN节点i的电压大小sy有一个取值范围
    u_j = {(j, t): 0 for j in range(E_BUS_NUM) for t in range(T)}  # ?DEN节点i的电压大小
    v_i = {(i, t): 0 for i in range(E_BUS_NUM) for t in range(T)}  # ?DEN中节点i的角度？直译可能不对，决策变量
    v_j = {(j, t): 0 for j in range(E_BUS_NUM) for t in range(T)}  # ?DEN中节点j的角度？直译可能不对
    # 约束
    # 7a
    for i in range(E_BUS_NUM):
        for j in range(E_BUS_NUM):
            if i != j:
                for t in range(T):
                    model.add_constraint(P_i[(i, t)] == model.sum(
                        x_ij[(i, j)] / (r_ij[(i, j)] ** 2 + x_ij[(i, j)] ** 2) * (v_i[(i, t)] - v_j[(j, t)]) + r_ij[
                            (i, j)] / (r_ij[(i, j)] ** 2 + x_ij[(i, j)] ** 2) * (u_i[(i, t)] - u_j[(j, t)])))
    # 7b
    for i in range(E_BUS_NUM):
        for j in range(E_BUS_NUM):
            if i != j:
                for t in range(T):
                    model.add_constraint(
                        P_i[(i, t)] == model.sum(
                            x_ij[(i, j)] / (r_ij[(i, j)] ** 2 + x_ij[(i, j)] ** 2) * (u_i[(i, t)] - u_j[(j, t)]) + r_ij[
                                (i, j)] / (r_ij[(i, j)] ** 2 + x_ij[(i, j)] ** 2) * (v_i[(i, t)] - v_j[(j, t)])))
    # 7c CHP+RES+BS_c+BS_d-EL-EC后1个是新增加的
    P_CHP_i = {(i, t): 0 for i in range(E_BUS_NUM) for t in range(T)}
    P_RES_i = {(i, t): 0 for i in range(E_BUS_NUM) for t in range(T)}
    P_BS_ci = {(i, t): 0 for i in range(E_BUS_NUM) for t in range(T)}
    P_BS_di = {(i, t): 0 for i in range(E_BUS_NUM) for t in range(T)}
    P_EL_i = {(i, t): 0 for i in range(E_BUS_NUM) for t in range(T)}
    P_EC_i = {(i, t): 0 for i in range(E_BUS_NUM) for t in range(T)}
    Q_CHP_i = {(i, t): 0 for i in range(E_BUS_NUM) for t in range(T)}
    H_CHP_j = {(i, t): 0 for i in range(E_BUS_NUM) for t in range(T)}
    Q_EL_i = {(i, t): 0 for i in range(E_BUS_NUM) for t in range(T)}
    for i in range(E_BUS_NUM):
        if i != 1:
            for t in range(T):
                model.add_constraint(P_i[(i, t)] == P_CHP_i[(i, t)] + P_RES_i[(i, t)] - P_BS_ci[(i, t)] + P_BS_di[(
                    i, t)] - P_EL_i[(i, t)] - P_EC_i[(i, t)])
    # 7d   RES中无功功率不考虑合理吗？+Q_RES_[(i,t)]
    for i in range(E_BUS_NUM):
        for t in range(T):
            model.add_constraint(Q_i[(i, t)] == Q_CHP_i[(i, t)] - Q_EL_i[(i, t)])
    # 7e
    for i in range(Line_NUM):
        for j in range(Line_NUM):
            for t in range(T):
                model.add_constraint(
                    P_ij[(i, j, t)] == (
                            r_ij[(i, j)] * (u_i[(i, t)] - u_j[(j, t)]) + x_ij[(i, j)] * (v_i[(i, t)] - v_j[(j, t)])) / (
                            r_ij[(i, j)] ** 2 + x_ij[(i, j)] ** 2))
    # 7f
    for i in range(Line_NUM):
        for j in range(Line_NUM):
            for t in range(T):
                model.add_constraint(Q_ij[(i, j, t)] == (x_ij[(i, j)] * (u_i[(i, t)] - u_j[(j, t)]) - r_ij[(i, t)] * (
                        v_i[(i, t)] - v_j[(j, t)])) / (r_ij[(i, j)] ** 2 + x_ij[(i, j)] ** 2))
    # 7g
    for i in range(Line_NUM):
        for j in range(Line_NUM):
            for t in range(T):
                model.add_constraint(-P_BUS_ij_MAX <= P_ij[(i, j, t)])
                model.add_constraint(P_BUS_ij_MAX >= P_ij[(i, j, t)])
    # 7h
    for i in range(E_BUS_NUM):
        for j in range(E_BUS_NUM):
            for t in range(T):
                model.add_constraint(u_BUS_MAX >= u_i[(i, t)])
                model.add_constraint(u_BUS_MIN <= u_i[(i, t)])
    ########################################(8)冷约束#######################################################################
    # 参数
    # k_EC, k_AC = 0.0016, 0.0024
    # EC_NUM, AC_NUM = 0, 0  # ? 之后可以改变数量，第一次实验先不考虑冷
    # P_EC_MAX, H_AC_MAX = 88.1, 168
    # C_EC_MAX, C_AC_MAX = 0, 0  # ?
    # # 决策变量
    # P_EC_n = {(n, t): model.continuous_var(lb=0, ub=P_EC_MAX,
    #                                        name=f'P_EC_{n}_{t}') for n in range(EC_NUM) for t in range(T)}
    # H_AC_n = {(n, t): model.continuous_var(lb=0, ub=H_AC_MAX,
    #                                        name=f'H_AC_{n}_{t}') for n in range(AC_NUM) for t in range(T)}
    # C_EC_n = {(n, t): model.continuous_var(lb=0, ub=C_EC_MAX,
    #                                        name=f'C_EC_{n}_{t}') for n in range(EC_NUM) for t in range(T)}
    # C_AC_n = {(n, t): model.continuous_var(lb=0, ub=C_AC_MAX,
    #                                        name=f'C_AC_{n}_{t}') for n in range(AC_NUM) for t in range(T)}

    # # 8a电制冷机
    # model.add_constraint(
    #     P_EC_n[(n, t)] <= P_EC_MAX for n in range(EC_NUM) for t in range(T))
    # model.add_constraint(
    #     P_EC_n[(n, t)] >= 0 for n in range(EC_NUM) for t in range(T))
    # model.add_constraint(C_EC_n[(n, t)] == a_EC_n*P_EC_n[(n, t)]
    #                      for n in range(EC_NUM) for t in range(T))
    # # 8b吸收式制冷机
    # model.add_constraint(
    #     H_AC_n[(n, t)] <= H_AC_MAX for n in range(AC_NUM) for t in range(T))
    # model.add_constraint(
    #     H_AC_n[(n, t)] >= 0 for n in range(AC_NUM) for t in range(T))
    # model.add_constraint(C_AC_n[(n, t)] == a_AC_n*H_AC_n[(n, t)]
    #                      for n in range(AC_NUM) for t in range(T))

    # # end input
    ######################################### (9)其他约束#####################################################################
    # 新引入对C_CHP的约束
    # ? I和J如何定义
    # d_C_CHP_n_P = {(n, t): 2 * a_n[5] * P_CHP_i[(n, t)] + a_n[3] * H_CHP_n[(n, t)] + a_n[2] for n in range(CHP_NUM) for
    #                t in range(t)}
    # d_C_CHP_n_H = {(n, t): 2 * a_n[5] * P_CHP_i[(n, t)] + a_n[3] * H_CHP_n[(n, t)] + a_n[2] for n in range(CHP_NUM) for
    #                t in range(t)}
    # model.add_constraint(
    #     C_CHP_n[(n, t)] >= d_C_CHP_n_P[(n, t)] * (P_CHP_n[(n, t)] - P_CHP_i[(n, t)]) + d_C_CHP_n_H[(n, t)] * (
    #             H_CHP_n[(n, t)] - H_CHP_j[(n, t)]) + C_CHP_n[(n, t)] for n in range(CHP_NUM) for t in range(t))

    # ......
    ################################### 目标函数##########################################################################
    # 定义目标函数
    obj_expr = W_set.E_PRICE[t] * P_i[(0, t)] + (model.sum(C_CHP_n[(n, t)] for n in range(CHP_NUM)) +
                                                 model.sum(C_BS_n[(n, t)] for n in range(BS_NUM)))

    # \+model.sum(k_EC*P_EC_n[(n, t)] for n in range(EC_NUM))+model.sum(
    #     k_AC*H_AC_n[(n, t)] for n in range(AC_NUM))  # todo:cool obj 先不考虑冷
    # 最小化目标函数
    model.minimize(obj_expr)
    # 返回t+1时刻的调度X, 目标值函数
    return model.solve_status()