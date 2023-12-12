from MILP import *
import matplotlib.pyplot as plt
from adp import *
from mpc import *
from sklearn.linear_model import LinearRegression


def real_data():
    """
    模拟真实环境信息

    Return:
        W_t: 24h真实外部环境信息
        R_t: 初始时刻真实资源状态信息
    """
    # 初始化24h的系统状态
    W_data = pd.read_excel('./data/W_data.xlsx', sheet_name='Sheet1')
    W_t = W()
    R_t = R()
    for t in range(T):
        W_t.E_PRICE[t] = W_data.loc[t, 'E_PRICE']
        W_t.P_EL[t] = W_data.loc[t, 'P_EL']
        W_t.H_TL[t] = W_data.loc[t, 'H_TL']
        W_t.C_TL[t] = W_data.loc[t, 'C_TL']
        W_t.P_RES[t] = W_data.loc[t, 'P_RES']
    R_t.E_TS[0] = 150
    R_t.E_BS[0] = 60
    return W_t, R_t


def algo_main(method='mpc'):
    # 生产真实系统状态
    W_t, R_t = real_data()
    if method == 'mpc':
        # MPC-3调度
        mpc_dispatch(W_t, R_t, n=6)
    elif method == 'adp':
        # ==========训练值函数==========
        # 模拟数据
        R_sets, W_sets = init_data()
        # 训练值函数
        il_adp(R_sets, W_sets)
        monotone_adp(R_sets, W_sets)
        # ==========ADP-IL实时调度==========
        # 实时调度
        adp_dispatch(W_t, R_t)


if __name__ == '__main__':
    algo_main()
