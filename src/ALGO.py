from MILP import *
import matplotlib.pyplot as plt


# 算法2
def il_adp(R_sets, W_sets):
    """IL算法"""

    print('----------------IL算法--------------------')
    train_samples = (R_sets[:N2], W_sets[:N2])
    # 读取solve_milp_all求解得到的R和X，聚类后作为专家经验
    solution_list = []
    for i in range(N2):
        solution_list.append(solve_milp_all(R_sets[i], W_sets[i]))
    # 初始化值函数
    for t in range(T):
        c_max = max(
            sum(solution[f'C_BS_{t_s}'] + solution[f'C_EP_{t_s}'] + solution[f'C_CHP_{t_s}'] for t_s in range(t, T))
            for solution in solution_list)
        for E_TS in range(E_TS_MAX):
            for E_BS in range(E_BS_MAX):
                v_table.value_table[E_TS, E_BS, t] = c_max
    for n, solution in enumerate(solution_list):
        r = train_samples[0][n]
        for t in range(T):
            print(f'epoch {n}:t={t} R{t}={{{r.E_TS[t]},{r.E_BS[t]}}} ==> {v_table.get_value(r, t)}')
            v = max(solution[f'C_BS_{t_s}'] + solution[f'C_EP_{t_s}'] + solution[f'C_CHP_{t_s}'] for t_s in range(t, T))
            v_table.update_r_count(r, t)
            alpha = 1 / v_table.get_r_count(r, t)
            z = alpha * v + (1 - alpha) * v_table.get_value(r, t)
            # 单调投影操作
            for t_v in range(T):
                r_v = (r.E_TS[t_v], r.E_BS[t_v])
                v_table.set_value(r, t_v, meth_func(z, r, t, r_v, t_v))
    # 清空计数
    for E_TS in range(E_TS_MAX + 1):
        for E_BS in range(E_BS_MAX + 1):
            v_table.r_count_table[E_TS, E_BS] = 0


def meth_func(z, r: R, t, r_v, t_v):
    """单调函数"""
    if r.E_BS[t] == r_v[0] and r.E_TS[t] == r_v[1]:
        return z
    elif r.E_BS[t] <= r_v[0] and r.E_TS[t] <= r_v[1]:
        return min(z, v_table.get_value2(r_v, t_v))
    elif r.E_BS[t] >= r_v[0] and r.E_TS[t] >= r_v[1]:
        return max(z, v_table.get_value2(r_v, t_v))
    else:
        return v_table.get_value2(r_v, t_v)


def monotone_adp(R_sets, W_sets):
    """adp算法"""

    print('----------------ADP算法--------------------')
    print('adp算法开始执行...')
    print(f'epoch总数: {N}')
    # R_sets, W_sets = init_data(mode='w')
    train_samples = (R_sets, W_sets)
    c_data = []
    for n in range(N):
        sample_index = n % SAMPLE_SIZE
        r = train_samples[0][sample_index]
        w = train_samples[1][sample_index]
        if n % 50 == 0:
            print(f'-------------------------epoch{n}-------------------------------')
        c_tmp = 0
        for t in range(T - 1):
            solution = solve_milp_v(r, w, t + 1)
            c_sum = solution['C_BS_t'] + solution['C_EP_t'] + solution['C_CHP_t']
            # c_sum = solution.objective_value
            c_tmp += c_sum
            v = solution.objective_value + np.random.randint(-10, 0)
            v_table.update_r_count(r, t)
            alpha = 1 / v_table.get_r_count(r, t)
            z = alpha * v + (1 - alpha) * v_table.get_value(r, t)
            # 单调投影操作
            for t_v in range(T):
                r_v = (r.E_TS[t_v], r.E_BS[t_v])
                v_table.set_value(r, t_v, meth_func(z, r, t, r_v, t_v))
            if n % 50 == 0:
                print(
                    f'epoch {n}:t={t + 1} R{t + 1}={{{r.E_TS[t + 1]},{r.E_BS[t + 1]}}} ==> {v_table.get_value(r, t + 1)}')
        c_data.append(c_tmp)
    print('adp算法执行完毕')
    # 绘制目标值训练趋势图
    plt.plot([i for i in range(N)], c_data)
    plt.show()

    # 画一下v的变化
    # v_data = []
    # for i in range(E_BS_MAX):
    #     v_data.append(v_table.get_value2((10, i), 10))
    #
    # plt.plot([i for i in range(E_BS_MAX)], v_data)
    # plt.show()


def dispatch():
    """
    实时调度
    """
    print('----------------ADP-IL调度--------------------')
    # 初始化24h的系统状态
    W_data = pd.read_excel('./data/W_data.xlsx', sheet_name='Sheet1')
    W_t = W()
    R_t = R()
    for t in range(T):
        W_t.E_PRICE[t] = W_data.loc[t, 'E_PRICE']
        W_t.P_EL = W_data.loc[t, 'P_EL']
        W_t.H_TL = W_data.loc[t, 'H_TL']
        W_t.C_TL = W_data.loc[t, 'C_TL']
        W_t.P_RES = W_data.loc[t, 'P_RES']
    R_t.E_BS[0] = np.random.uniform(E_BS_MIN, E_BS_MAX)
    R_t.E_TS[0] = np.random.uniform(E_TS_MIN, E_TS_MAX)
    print('系统状态初始化完毕！开始实时调度……')
    # 输入状态，输出调度
    for t in range(T - 1):
        # w模式，保存t时刻的调度结果到Dispatch文件夹下
        solution = solve_milp_v(R_t, W_t, t + 1, mode='w', data_path=f'./data/Dispatch/dispatch_result_{t + 1}.xlsx')
        print(f't={t + 1}时刻调度结果求解成功！')


def algo_main():
    # 模拟数据
    R_sets, W_sets = init_data()
    # 训练值函数
    il_adp(R_sets, W_sets)
    monotone_adp(R_sets, W_sets)
    # 实时调度
    dispatch()


if __name__ == '__main__':
    algo_main()
