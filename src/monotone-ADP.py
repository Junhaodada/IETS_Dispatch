"""
Algorithm 1. off-line pre-learning process of monotone-ADP(small size IETS)

Small size IETS is with 6 electricity buses and 5 thermal nodes, 1 BS, and 1 TS.

"""
import numpy as np
import pandas as pd

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
    def __init__(self, t, R:R) -> None:
        self.t = t # t time-slot(1-24h)
        self.R = R
        self.value = 0

class X:
    """
    resource state `Rt` includes E_TS and E_BS(Energy stored in TS/BS at time-slot t)
    """
    def __init__(self, t, R:R) -> None:
        self.t = t # t time-slot(1-24h)
        self.R = R

# Step1: generate a set of training samples Ω,containing trajectories of exogenous information
# generate a set of training samples Ω
N = 2500
T = 24
SAMPLE_SIZE = 1 # 1000
E_TS_MAX = 200 # KWh
E_BS_MAX = 80 # KWh
R_set = []
W_set = []
S_set = []
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
# V = np.zeros((T,N))

# for n in range(1, N):
#     s_c = S[0] # choose a sample
#     for t in range(1, T + 1):
#         pass

