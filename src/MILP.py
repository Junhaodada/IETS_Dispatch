from docplex.mp.model import Model
from ADP import X,V,S,W,R,C,E_TS_MAX,E_BS_MAX
from utils import *
from typing import List
def solve_milp(s_c:List[S], t):
    """
    solve (20a) MILP with cplex

    Parameter
    ---------
    s_c: List[S]
        s_c是算法1选择的样本, 包含{W, S}, W为外部环境, S为资源状态
        举例: 
            s_c[t] 代表t时刻的S, s_c[t].R 代表t时刻的R
            s_c[t].R 代表t时刻的R
            s_c[t].R 代表t时刻的R
    t: int
        t是算法1循环迭代的时刻
    """
    # R_t_minus_1:R = s_c[t-1].R # t-1时刻的R
    # R_t:R = s_c[t].R # t时刻的R

 
    # 创建模型
    model = Model(name='IETS MILP MODEL')

    # 定义决策变量
    # 变量相关参数
    T = 24  # time num
    CHP_NUM = 1  # CHP num
    RES_NUM = 1 # RES num
    BS_NUM = 1 # BS num
    TS_NUM = 1 # TS num
    E_BUS_NUM = 6 # electricity buses num
    T_NODE_NUM = 5 # thermal nodes num
    SAMPLE_SIZE = 1 # 1000
    E_TS_MAX = 200 # KWh
    E_BS_MAX = 80 # KWh
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
    G_NUM = 0
    ##########################################决策变量###############################################
    # 模型中所有约束会使用到以下变量，与论文中的X_t完全对应，没有定义X_t_A，编写各个约束时自行添加即可
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
    # m_TDN
    # tem_TDN
    # u_i
    # P_ij
    # Q_ij
    gama_g = [model.continuous_var(lb=0,ub=1) for _ in range(G_NUM)]
    # Xt_A ...
    # X_t 包含的所有决策变量, 需要把上述所有子变量加入到X_t中，作为返回值
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
            # X_t_A ...
        }

    ###################################目标函数##########################################################################
    # 定义目标函数
    # $X_t=argmin_{X_t \in \Pi _t} (C_t(S_t,X_t)+\sum_{g\in G}\gamma _g V_t^x(R_g))$
    obj_expr = [cal_cost(s_c[t],X_t[t])+ (model.sum(gama_g[g]) for g in range(G_NUM)) for t in range(T)]
    # 最小化目标函数
    model.minimize(obj_expr)
    #####################################模型约束#######################################################################
    ####################################(1)TS约束#######################################################################
    # 补充的决策变量
    m_TS_MIN, m_TS_MAX = 0, 0
    tem_TS_MIN, tem_TS_MAX = 20, 100
    H_TS_MIN, H_TS_MAX = 0, 0
    m_TS_n = {(n, t): model.continuous_var(lb=m_TS_MIN, ub=m_TS_MAX, name=f'm_TS_{n}_{t}') for n in range(TS_NUM) for t in range(T)}
    tem_TS_n = {(n, t): model.continuous_var(lb=tem_TS_MIN, ub=tem_TS_MAX, name=f'tem_TS_{n}_{t}') for n in range(TS_NUM) for t in range(T)}
    tem_TS_S_n = {(n, t): model.continuous_var(lb=tem_TS_MIN, ub=tem_TS_MAX, name=f'tem_TS_S_{n}_{t}') for n in range(TS_NUM) for t in range(T)}
    tem_TS_R_n = {(n, t): model.continuous_var(lb=tem_TS_MIN, ub=tem_TS_MAX, name=f'tem_TS_S_{n}_{t}') for n in range(TS_NUM) for t in range(T)}
    H_TS_n = {(n, t): model.continuous_var(lb=H_TS_MIN, ub=H_TS_MAX, name=f'H_TS_{n}_{t}') for n in range(TS_NUM) for t in range(T)}
    # 补充的约束
    # 1a
    model.add_constraint(m_TS_n[(n,t)] == 0.8*a_TS_cn[(n,t)] -0.8*a_TS_dn[(n,t)] for n in range(TS_NUM) for t in range(T))
    # 1b
    model.add_constraint(H_TS_cn[(n,t)]>=a_TS_cn[(n,t)]*0 for n in range(TS_NUM) for t in range(T))
    model.add_constraint(H_TS_cn[(n,t)]<=a_TS_cn[(n,t)]*100 for n in range(TS_NUM) for t in range(T))
    # 1c
    model.add_constraint(H_TS_dn[(n,t)]>=a_TS_dn[(n,t)]*0 for n in range(TS_NUM) for t in range(T))
    model.add_constraint(H_TS_dn[(n,t)]<=a_TS_dn[(n,t)]*100 for n in range(TS_NUM) for t in range(T))
    # 1d
    model.add_constraint(a_TS_cn[(n,t)]+a_TS_dn[(n,t)]<=1 for n in range(TS_NUM) for t in range(T))
    # 1f
    model.add_constraint(s_c[t].R.E_TS == (1-0.95)*s_c[t].R.E_TS - H_TS_dn[(n,t)]+0.95*H_TS_cn[(n,t)] for n in range(TS_NUM) for t in range(T))
    # 1g
    C_wt, p_water, V_TS_n, tem_TS = 4.182/3600, 1000, 2.46, 30
    model.add_constraint(s_c[t].R.E_TS==C_wt*p_water*V_TS_n*(tem_TS_n[(n,t)]-tem_TS) for n in range(TS_NUM) for t in range(T))
    # 1h
    model.add_constraint(s_c[t].R.E_TS>=0 for n in range(TS_NUM) for t in range(T))
    model.add_constraint(s_c[t].R.E_TS<=E_TS_MAX for n in range(TS_NUM) for t in range(T))
    # 1i
    model.add_constraint(tem_TS_n[(n,t)]>=0 for n in range(TS_NUM) for t in range(T))
    model.add_constraint(tem_TS_n[(n,t)]<=tem_TS_MAX for n in range(TS_NUM) for t in range(T))
    # 1j
    model.add_constraint(tem_TS_S_n[(n,t)]>=a_TS_cn[(n,t)]*tem_TS_n[(n,t)] for n in range(TS_NUM) for t in range(T))
    model.add_constraint(tem_TS_R_n[(n,t)]>=a_TS_cn[(n,t)]*tem_TS_n[(n,t)] for n in range(TS_NUM) for t in range(T))
    model.add_constraint(tem_TS_S_n[(n,t)]<=a_TS_dn[(n,t)]*tem_TS_n[(n,t)]+(1-a_TS_dn[(n,t)])*tem_TS_MAX for n in range(TS_NUM) for t in range(T))
    model.add_constraint(tem_TS_R_n[(n,t)]<=a_TS_dn[(n,t)]*tem_TS_n[(n,t)]+(1-a_TS_dn[(n,t)])*tem_TS_MAX for n in range(TS_NUM) for t in range(T))
    # 1k
    model.add_constraint(H_TS_dn[(n,t)]==a_TS_dn[(n,t)]*0.8*C_wt*(tem_TS_S_n[(n,t)]-tem_TS_R_n[(n,t)]) for n in range(TS_NUM) for t in range(T))
    # 1l
    model.add_constraint(H_TS_cn[(n,t)]==a_TS_cn[(n,t)]*0.8*C_wt*(tem_TS_S_n[(n,t)]-tem_TS_R_n[(n,t)]) for n in range(TS_NUM) for t in range(T))
    # 1m
    model.add_constraint(H_TS_n[(n,t)]==H_TS_cn[(n,t)]-H_TS_dn[(n,t)] for n in range(TS_NUM) for t in range(T))
    # 1n
    model.add_constraint(tem_TS_S_n[(n,t)] >= 20 for n in range(TS_NUM) for t in range(T))
    model.add_constraint(tem_TS_S_n[(n,t)] <= 100 for n in range(TS_NUM) for t in range(T))    
    model.add_constraint(tem_TS_R_n[(n,t)] >= 20 for n in range(TS_NUM) for t in range(T))
    model.add_constraint(tem_TS_R_n[(n,t)] <= 100 for n in range(TS_NUM) for t in range(T))
    # 1o
    model.add_constraint(s_c[1].R.E_TS==s_c[T].R.E_TS)
    ##########################################(2)CHP约束##################################################################
    # 2f
    # model.add_constraint( H_CHP_n[(n,t)]== for n in range(TS_NUM) for t in range(T))
    # 2g
    # 2h
    ###########################################(3)热负载约束###############################################################
    # input your code


    # end input
    ###########################################(4)TDN/热配网约束###########################################################
    # input your code


    # end input
    ############################################(5)RES约束#################################################################
    # input your code


    # end input
    ###########################################(6)BS约束####################################################################
    # input your code


    # end input
    ##########################################(7)EDN/电配网约束##############################################################
    # input your code


    # end input
    #########################################(8)冷约束#######################################################################
    # input your code


    # end input
    #########################################(9)其他约束#####################################################################
    # gama_g约束
    model.add_constraint(model.sum(gama_g)==1)
    # 状态转移约束
    # ......

    # 返回t+1时刻的调度X, 目标值函数
    return X_t[t+1], model.get_solve_details()

