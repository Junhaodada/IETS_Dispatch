import numpy as np
import pandas as pd
from typing import List
from params import *
from MILP import *

# 生成样本
R_sets: List[R] = []
SAMPLE_SIZE = 1  # ? 1000
for i in range(SAMPLE_SIZE):
    R_set = R()
    for t in range(T):
        for n in range(TS_NUM):
            R_set.E_TS[(n, t)] = np.random.uniform(E_TS_MIN, E_TS_MAX)
        for n in range(BS_NUM):
            R_set.E_BS[(n, t)] = np.random.uniform(E_BS_MIN, E_BS_MAX)
    R_sets.append(R_set)

W_sets: List[W] = []
W_data = pd.read_excel('W_data.xlsx', sheet_name='Sheet1')
# 热负载比例
TL_LOAD_RATIO = [0, 0, 0.3, 0.3, 0.4]
for i in range(SAMPLE_SIZE):
    W_set = W()
    for t in range(T):
        W_set.E_PRICE[t] = W_data.loc[t, 'E_PRICE']
        W_set.TEM_AM = W_data.loc[t, 'TEM_AM']
        for i in range(E_BUS_NUM):
            W_set.P_EL[(i, t)] = W_data.loc[t, 'P_EL']
            W_set.Q_EL[(i, t)] = W_data.loc[t, 'Q_EL']

        for n in range(TL_NUM):
            W_set.H_TL[(n, t)] = W_data.loc[t, 'H_TL'] * TL_LOAD_RATIO[n]
        for n in range(RES_NUM):
            W_set.P_RES[(i, t)] = W_data.loc[t, 'P_RES']
            W_set.Q_RES[(i, t)] = W_data.loc[t, 'Q_RES']
    W_sets.append(W_set)

print(R_sets[0].E_BS[(0, 0)])
print(W_sets[0].H_TL[0, 0])
print(W_sets[0].H_TL[1, 0])

# 测试模型
# 使用随机生成的第一个样本，求解t=1时的调度
print(solve_milp(W_sets[0], R_sets[0], 1))
