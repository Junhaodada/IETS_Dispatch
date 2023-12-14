import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_balance_fig(type):
    for j in range(5):
        """ 处理数据 """
        data_1 = pd.read_excel(
            f'./data/Solution/Solution_{j}.xlsx', header=None)
        data_2 = pd.read_excel('./data/W_data.xlsx')
        eta_he = 0.98
        a_EC = 4
        a_AC = 0.7
        P_EL = np.array(data_2.loc[:, 'P_EL'])
        H_TL = np.array(data_2.loc[:, 'H_TL']) / eta_he
        C_TL = np.array(data_2.loc[:, 'C_TL'])

        columns = ['P_RES', 'P_PG', 'P_BS_d', 'P_CHP', 'P_EC', 'P_BS_c', 'P_EL',
                   'H_AC', 'H_TL', 'H_TS_c', 'H_TS_d', 'H_CHP',
                   'H_AC', 'P_EC', 'C_TL']
        data_dict = {}
        for column_name in columns:
            if column_name not in ['P_EL', 'H_TL', 'C_TL']:
                rows = data_1[data_1[0].str.startswith(column_name)]
                if column_name == 'H_AC':
                    data = np.array(rows[1]) * a_AC
                elif column_name == 'P_EC':
                    data = np.array(rows[1]) * a_EC
                else:
                    data = np.array(rows[1])
                data_dict[column_name] = data

        data_dict['P_EL'] = P_EL
        data_dict['H_TL'] = H_TL
        data_dict['C_TL'] = C_TL

        labels = []
        for label in columns:
            col = label.split('_')
            l = len(col)
            if l == 2:
                s = '$' + col[0] + '^' + '{' + col[1] + '}' + '$'
            else:
                s = '$' + col[0] + '^' + '{' + col[1] + \
                    '}' + '_' + '{' + col[2] + '}' + '$'
            labels.append(s)

        """ 柱状图 """
        x = np.arange(24)
        width = 0.8
        fig, ax1 = plt.subplots()
        # bottom = data_dict[columns[0]]
        # for i in range(4):
        bottom = data_dict['P_RES']
        for i, column in enumerate(columns):
            # Electric
            # if column in ['P_RES', 'P_PG', 'P_BS_d', 'P_CHP']:
            # # Heat
            # if column in ['H_AC', 'H_TL', 'H_TS_c']:
            # Cool
            # if column in ['H_AC', 'P_EC']:
            # All
            if column not in ['P_EC', 'P_BS_c', 'P_EL', 'H_TS_d', 'H_CHP', 'C_TL']:
                # x 轴上方
                if column == 'P_RES':
                    rects = ax1.bar(
                        x, data_dict[columns[i]], width, label=labels[i], zorder=10)
                else:
                    rects = ax1.bar(
                        x, data_dict[columns[i]], width, label=labels[i], bottom=bottom, zorder=10)
                    bottom += data_dict[columns[i]]

        # for i in range(8, 17):
        bottom = -data_dict['P_EC']
        for i, column in enumerate(columns):
            # Electric
            # if column in ['P_EC', 'P_BS_c', 'P_EL']:
            # Heat
            # if column in ['H_TS_d', 'H_CHP']:
            # Cool
            # if column in ['C_TL']:
            # All
            if column in ['P_EC', 'P_BS_c', 'P_EL', 'H_TS_d', 'H_CHP', 'C_TL']:
                # x 轴下方
                if column == 'P_EC':
                    # if column == 'H_TS_d':
                    # if column == 'C_TL':
                    rects = ax1.bar(
                        x, -data_dict[columns[i]], width, label=labels[i], zorder=10)
                else:
                    rects = ax1.bar(
                        x, -data_dict[columns[i]], width, label=labels[i], bottom=bottom, zorder=10)
                    bottom -= data_dict[columns[i]]

        ax1.axhline(0, color='black', linewidth=0.5)
        ax1.set_xlabel('Time (h)')
        ax1.set_ylabel('Values (kW)')
        title_name = ['Electric', 'Heat', 'Cold']
        # ax1.set_title(f'Conservation of {title_name[2]}')
        ax1.set_title(f'Conservation of Electric Heat Cold')
        ax1.set_xticks(x)
        ax1.set_ylim(-900, 700)
        ax1.tick_params(axis='x', labelsize=8)
        handles1, labels1 = ax1.get_legend_handles_labels()
        upper_num, lower_num = 8, 8
        legend_upper = ax1.legend(handles1[:upper_num], labels1[:upper_num],
                                  loc='upper left', bbox_to_anchor=(0, 1), prop={'size': 4}, ncol=2)
        legend_lower = ax1.legend(handles1[lower_num:], labels1[lower_num:],
                                  loc='lower left', bbox_to_anchor=(0, 0), prop={'size': 4}, ncol=2)
        ax1.add_artist(legend_upper)
        ax1.add_artist(legend_lower)
        ax1.grid(True, zorder=0, color='lightgrey', linewidth=0.5)
        plt.savefig(f'./data/figure/solution_{j}.jpg', dpi=300)
        # plt.show()


if __name__ == "__main__":
    plot_balance_fig('All')
