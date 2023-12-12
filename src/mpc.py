import matplotlib.pyplot as plt
import pandas as pd

from MILP import *


def mpc_dispatch(W_t: W, R_t: R, n=3):
    """
    MPC-n
    模型预测控制-调度算法

    Args:
        n: 预测n步未来场景
    """
    cost = []
    E_TS = []
    E_BS = []
    # 预测环境信息
    for t in range(T-1):
        print(f't={t}调度完成')
        t_start = t
        t_end = t + n
        if t_end > T:
            t_end = T
        solution = solve_milp_mpc(R_t, W_t, t_start, t_end)
        R_t.E_TS[t+1] = solution[f'E_TS_{t+1}']
        R_t.E_BS[t+1] = solution[f'E_BS_{t+1}']
        # 保存t时目标值
        cost.append(solution.objective_value)
        # 保存t时能量，绘制能量变化曲线
        E_TS.append(solution[f'E_TS_{t}'])
        E_BS.append(solution[f'E_BS_{t}'])
    E_TS.append(solution[f'E_TS_{t+1}'])
    E_BS.append(solution[f'E_BS_{t+1}'])
    plt.plot([_ for _ in range(T-1)], cost)
    plt.show()
    pd.DataFrame({'E_TS': E_TS, 'E_BS': E_BS}).to_excel('./data/mpc_energy.xlsx', index=False)
