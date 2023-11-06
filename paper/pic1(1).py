"""
x轴上方
P_BS_c
P_RES
P_CHP
P_PG
x轴下方
P_EC
P_BS_d
W_set.P_EL[t]

每个数据的格式为 df['P_EC'] row*column：24*1
使用python画一个t从1到24小时的条形图
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def Visual(P_BS_c, P_RES, P_CHP, P_PG, P_EC, P_BS_d, P_EL):
    x = np.arange(24)  
    width = 0.8  
    fig, ax1 = plt.subplots()
    rects1 = ax1.bar(x, P_BS_c, width, label='$P^{BS}_{c}$')
    rects2 = ax1.bar(x, P_RES, width, label='$P^{RES}$', bottom = P_BS_c)
    rects3 = ax1.bar(x, P_CHP, width, label='$P^{CHP}$', bottom = np.array(P_BS_c) + np.array(P_RES))
    rects4 = ax1.bar(x, P_PG, width, label='$P^{PG}$', bottom = np.array(P_CHP) + np.array(P_BS_c) + np.array(P_RES))
    
    rects5 = ax1.bar(x, -np.array(P_EC), width, label='$P^{EC}$')
    rects6 = ax1.bar(x, -np.array(P_BS_d), width, label='$P^{BS}_{d}$', bottom = -np.array(P_EC))
    rects7 = ax1.bar(x, -np.array(P_EL), width, label='$P^{EL}$', bottom = -np.array(P_BS_d)-np.array(P_EC))

    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.set_xlabel('小时 (t)')
    ax1.set_ylabel('值')
    ax1.set_ylim(-350, 450)
    ax1.tick_params(axis='y', labelsize=8)
    ax1.set_title('电量')
    ax1.set_xticks(x)
    ax1.legend()
    ax1.tick_params(axis='x', labelsize=8)
    plt.grid(color='lightgray')
    plt.gca().set_axisbelow(True)

    handles1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(handles1, labels1, loc='upper left', prop={'size': 6},  ncol = 2)
    plt.savefig(f'./Result/pic.jpg', dpi = 300)
    
    plt.show()
    
    
if __name__ == '__main__':
    data = pd.read_excel('./data/pic.xlsx')
    P_BS_c = data.loc[:, 'P_BS_c'].tolist()
    P_RES = data.loc[:, 'P_RES'].tolist()
    P_CHP = data.loc[:, 'P_CHP'].tolist()
    P_PG = data.loc[:, 'P_PG'].tolist()
    P_EC = data.loc[:, 'P_EC'].tolist()
    P_BS_d = data.loc[:, 'P_BS_d'].tolist()
    P_EL = data.loc[:, 'P_EL'].tolist()
    Visual(P_BS_c, P_RES, P_CHP, P_PG, P_EC, P_BS_d, P_EL)
    
    



