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
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 从Excel文件中读取数据
data = pd.read_excel('./data/pic.xlsx')

# 提取上方和下方的数据列
upper_columns = ['P_BS_c', 'P_RES', 'P_CHP', 'P_PG']
lower_columns = ['P_EC', 'P_BS_d', 'P_EL']

# 提取 t 值
t = data['t']

# 提取上方和下方的数据列的值
upper_values = [
    data['P_BS_c'],
    data['P_RES'],
    data['P_CHP'],
    data['P_PG'],
]
lower_values = [
    data['P_EC'],
    data['P_BS_d'],
    data['P_EL'],
]

# 创建一个 t 从 1 到 24 小时的柱状图
plt.figure(figsize=(12, 6))  # 设置图形大小

# 定义颜色列表，用于上下两组数据的柱形图
upper_colors = ['b', 'g', 'r', 'c']
lower_colors = ['m', 'y', 'k']

# 绘制上方的数据列
for i, column in enumerate(upper_columns):
    plt.bar(t, upper_values[i], label=column, alpha=0.7, width=0.5, align='center', color=upper_colors[i])

# 绘制下方的数据列
for i, column in enumerate(lower_columns):
    plt.bar(t, -1 * lower_values[i], label=column, alpha=0.7, width=0.5, align='center', color=lower_colors[i])

# 设置 x 轴和 y 轴标签
plt.xlabel('小时 (t)')
plt.ylabel('值')

# 添加图例
plt.legend()

# 显示图形
plt.grid(True)
plt.title('电量')
plt.show()
