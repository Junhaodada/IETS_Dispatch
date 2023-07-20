"""
Algorithm 1. off-line pre-learning process of monotone-ADP(small size IETS)

Small size IETS is with 6 electricity buses and 5 thermal nodes, 1 BS, and 1 TS.

"""
import numpy as np
import pandas as pd
from typing import List
from MILP import solve_milp
from utils import cal_cost

class W:
    """
    Exogenous information `W_t` of IETS includes stochastic processes of RES generation, 
    ambient temperature,electrical and thermal loads, and real-time price.
    """
    def __init__(self, t, E_PRICE, P_EL, Q_EL, H_TL, TEM_AM, P_RES, Q_RES) -> None:
        self.t = t # t time-slot(1-24h)
        self.E_PRICE = E_PRICE  # electrical real-time price
        self.P_EL = P_EL # active electrical load 
        self.Q_EL = Q_EL # reactive electrical load
        self.H_TL = H_TL # thermal load in node
        self.TEM_AM = TEM_AM # ambient temperature
        self.P_RES = P_RES # Upper active power output limit of RES
        self.Q_RES = Q_RES # Upper reactive power output limit of RES
    
    def __str__(self) -> str:
        return f'W{self.t}:<E_PRICE:{self.E_PRICE},P_EL:{self.P_EL},Q_EL:{self.Q_EL},H_TL:{self.H_TL},TEM_AM:{self.TEM_AM},P_RES:{self.P_RES},Q_RES:{self.Q_RES}>'

class R:
    """
    resource state `Rt` includes E_TS and E_BS(Energy stored in TS/BS at time-slot t)
    """
    def __init__(self, t = 0, E_TS = 0, E_BS = 0) -> None:
        self.t = t # t time-slot(1-24h)
        self.E_TS = E_TS # 0-200 kWh
        self.E_BS = E_BS # 0-80 kWh
    
    def __str__(self) -> str:
        return f'R{self.t}:<E_TS:{self.E_TS},E_BS:{self.E_BS}'

class S:
    """
    System state `S_t={W_t, R_t}` includes exogeneous information 
    and resource state R_t where
    """
    def __init__(self, t, W:W, R:R) -> None:
        self.t = t
        self.W = W
        self.R = R
    
    def __str__(self) -> str:
        return f'S{self.t}:<{self.W},{self.R}>'

class V:
    """
    value function of t
    """
    def __init__(self, t, R = None) -> None:
        self.t = t # t time-slot(1-24h)
        self.R = R
        self.value = 0

class X:
    """
    X_t
    """
    def __init__(self, t) -> None:
        self.t = t # t time-slot(1-24h)
        # self.H_CHP_n_t = [0 for i in range(CHP_NUM)]
        self.H_CHP_n_t= [0 for n in range(CHP_NUM)],
        self.P_CHP_n_t= [0 for n in range(CHP_NUM)],
        self.a_BS_dn_t= [0 for n in range(BS_NUM)],
        self.a_BS_cn_t= [0 for n in range(BS_NUM)],
        self.a_TS_dn_t= [0 for n in range(TS_NUM)],
        self.a_TS_cn_t= [0 for n in range(TS_NUM)],
        self.H_TS_dn_t= [0 for n in range(TS_NUM)],
        self.H_TS_cn_t= [0 for n in range(TS_NUM)],
        self.P_BS_dn_t= [0 for n in range(BS_NUM)],
        self.P_BS_cn_t= [0 for n in range(BS_NUM)],
        self.P_RES_n_t= [0 for n in range(RES_NUM)],
        self.Q_RES_n_t= [0 for n in range(RES_NUM)]

class C:
    """
    C_t cost function
    """
    def __init__(self, S:S,X:X) -> None:
        self.S = S
        self.X = X
        self.value = 0

# Step1: generate a set of training samples Ω,containing trajectories of exogenous information
# generate a set of training samples Ω
N = 2500
T = 24  # 时间步数
CHP_NUM = 1  # CHP设备数量
RES_NUM = 1 # RES设备数量
BS_NUM = 1 # BS设备数量
TS_NUM = 1 # TS设备数量
E_BUS_NUM = 6 # electricity buses
T_NODE_NUM = 5 # thermal nodes
SAMPLE_SIZE = 1 # 1000
E_TS_MAX = 200 # KWh
E_BS_MAX = 80 # KWh
ALPHA = 0.1
INF = 1e9
R_set:List[List[R]] = []
W_set:List[List[W]] = []
S_set:List[List[S]] = []
## generate R,W,S set
w_data = pd.read_excel('src/W_data.xlsx',sheet_name='Sheet1')
for s in range(SAMPLE_SIZE):
    r_tmp, w_tmp, s_tmp = [], [], []
    for t in range(1, T + 1):
        r = R(t, np.random.uniform(0,E_TS_MAX,1)[0], np.random.uniform(0,E_BS_MAX,1)[0])
        w = W(t,E_PRICE=w_data.loc[t-1,'E_PRICE'],P_EL=w_data.loc[t-1,'P_EL'],
                       Q_EL=w_data.loc[t-1,'Q_EL'],H_TL=w_data.loc[t-1,'H_TL'],
                       TEM_AM=w_data.loc[t-1,'TEM_AM'],P_RES=w_data.loc[t-1,'P_RES'],Q_RES=w_data.loc[t-1,'Q_RES'])
        s = S(t,r,w)
        r_tmp.append(r)
        w_tmp.append(w)
        s_tmp.append(s)
    R_set.append(r_tmp)
    W_set.append(w_tmp)
    S_set.append(s_tmp)


print(S_set[0][0].R,S_set[0][0].W)
# V set
V_set:List[List[V]]=[[] for i in range(SAMPLE_SIZE)]
for i in range(SAMPLE_SIZE):
    for t in range(T):
        V_set[t].append(V(t))

for t in range(T):
        V_set[0][t].R = R_set[0][t]
X_set = []
for n in range(1, N): # step2: ...
    # step3: choose a sample
    s_c = S_set[n]
    for t in range(1, T + 1):
        # step4: ...
        X_t_plus_1 = solve_milp(s_c, t)

        V_t_plus_1_n_minus_1 = V_set[n][t]
        R_t_plus_1_n = s_c[t+1].R
        R_t_plus_1_n.R.E_BS = 0
        R_t_plus_1_n.R.E_TS = 0
        V_t_plus_1_n_minus_1.R = R_t_plus_1_n
        V_t_plus_1_n_minus_1.calValue()

        v_t_n = V(t)
        v_t_n.R = s_c[t].R[t]
        v_t_n.value = min(cal_cost(s_c[t+1],X_t_plus_1)+V_t_plus_1_n_minus_1)

        # step5: ...
        z_t_n = V_set[n][t]
        z_t_n.R = s_c[t].R
        z_t_n.value = ALPHA * v_t_n.value + (1 - ALPHA) * V_set[n][t].value

        # step6: ...



# algorithm 2 off-line pre-learning process of ADP-IL

# 专家经验
X_set:List[List[X]] = []
S_set:List[List[S]] = []

expert_demonstrantions_S = S_set[:5]
expert_demonstrantions_X = X_set[:5]
expert_demonstrantions = [(expert_demonstrantions_S[i],expert_demonstrantions_X[i]) for i in range(SAMPLE_SIZE)]

V_set:List[List[V]]=[[] for i in range(SAMPLE_SIZE)]
for i in range(SAMPLE_SIZE):
    for t in range(T):
        V_set[t].append(V(t))
for t in range(T):
        V_set[0][t].R = R_set[0][t]
        V_set[0][t].value = INF


n = 1
expert_tag=[_ for _ in range(5)]
while len(expert_tag):
    e = expert_demonstrantions[0]
    for t in range(1, T):
        pass

    n+=1



if __name__ == "__main__":
    pass