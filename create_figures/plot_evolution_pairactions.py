import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.interpolate import interp1d

game_type = 'last_action'
C_ratio = np.load(f'outputs/{game_type}/round_wise_data/rounds_total_C_ratio_per_step_20_20_10_0.25.npy')
pair_actions = np.load(f'outputs/{game_type}/round_wise_data/rounds_total_pair_actions_20_20_10_0.25.npy', allow_pickle=True)

# # rerun the 8th round for removing the outlier
# C_ratio = pd.read_csv('NR_C_ratio-modified.csv')
# # change the pair actions for the 8th round to this array: [{'CC': 11, 'CD': 11, 'DC': 15, 'DD': 7}, {'CC': 11, 'CD': 8, 'DC': 11, 'DD': 14}, {'CC': 18, 'CD': 4, 'DC': 2, 'DD': 20}, {'CC': 30, 'CD': 2, 'DC': 4, 'DD': 8}, {'CC': 32, 'CD': 2, 'DC': 4, 'DD': 6}, {'CC': 34, 'CD': 1, 'DC': 3, 'DD': 6}, {'CC': 35, 'CD': 1, 'DC': 2, 'DD': 6}, {'CC': 36, 'CD': 2, 'DC': 2, 'DD': 4}, {'CC': 38, 'CD': 0, 'DC': 2, 'DD': 4}, {'CC': 39, 'CD': 1, 'DC': 1, 'DD': 3}, {'CC': 41, 'CD': 1, 'DC': 0, 'DD': 2}, {'CC': 43, 'CD': 0, 'DC': 0, 'DD': 1}, {'CC': 44, 'CD': 0, 'DC': 0, 'DD': 0}, {'CC': 44, 'CD': 0, 'DC': 0, 'DD': 0}, {'CC': 44, 'CD': 0, 'DC': 0, 'DD': 0}, {'CC': 44, 'CD': 0, 'DC': 0, 'DD': 0}, {'CC': 44, 'CD': 0, 'DC': 0, 'DD': 0}, {'CC': 44, 'CD': 0, 'DC': 0, 'DD': 0}, {'CC': 44, 'CD': 0, 'DC': 0, 'DD': 0}, {'CC': 44, 'CD': 0, 'DC': 0, 'DD': 0}]
# pair_actions = np.load('outputs/network_ratio/round_wise_data/rounds_total_pair_actions_20_20_10_0.25.npy', allow_pickle=True)
# pair_actions[7] = [{'CC': 11, 'CD': 11, 'DC': 15, 'DD': 7}, {'CC': 11, 'CD': 8, 'DC': 11, 'DD': 14}, {'CC': 18, 'CD': 4, 'DC': 2, 'DD': 20}, {'CC': 30, 'CD': 2, 'DC': 4, 'DD': 8}, {'CC': 32, 'CD': 2, 'DC': 4, 'DD': 6}, {'CC': 34, 'CD': 1, 'DC': 3, 'DD': 6}, {'CC': 35, 'CD': 1, 'DC': 2, 'DD': 6}, {'CC': 36, 'CD': 2, 'DC': 2, 'DD': 4}, {'CC': 38, 'CD': 0, 'DC': 2, 'DD': 4}, {'CC': 39, 'CD': 1, 'DC': 1, 'DD': 3}, {'CC': 41, 'CD': 1, 'DC': 0, 'DD': 2}, {'CC': 43, 'CD': 0, 'DC': 0, 'DD': 1}, {'CC': 44, 'CD': 0, 'DC': 0, 'DD': 0}, {'CC': 44, 'CD': 0, 'DC': 0, 'DD': 0}, {'CC': 44, 'CD': 0, 'DC': 0, 'DD': 0}, {'CC': 44, 'CD': 0, 'DC': 0, 'DD': 0}, {'CC': 44, 'CD': 0, 'DC': 0, 'DD': 0}, {'CC': 44, 'CD': 0, 'DC': 0, 'DD': 0}, {'CC': 44, 'CD': 0, 'DC': 0, 'DD': 0}, {'CC': 44, 'CD': 0, 'DC': 0, 'DD': 0}]

# data = pd.DataFrame(pair_actions)
# data = pd.DataFrame.from_dict(pair_actions)
# data = pd.DataFrame.from_records(pair_actions)
scenario = 'LA+NR'


CC_list = []
CD_list = []
DC_list = []
DD_list = []

for i in range(np.shape(pair_actions)[0]):
    CC_list_tem = []
    CD_list_tem = []
    DC_list_tem = []
    DD_list_tem = []
    for j in range(np.shape(pair_actions)[1]):
        CC_list_tem.append(pair_actions[i,j]['CC'])
        CD_list_tem.append(pair_actions[i,j]['CD'])
        DC_list_tem.append(pair_actions[i,j]['DC'])
        DD_list_tem.append(pair_actions[i,j]['DD'])
    CC_list.append(CC_list_tem)
    CD_list.append(CD_list_tem)
    DC_list.append(DC_list_tem)
    DD_list.append(DD_list_tem)

x = np.arange(1, 21)

total_number = np.sum(CC_list + CD_list + DC_list + DD_list, axis=0) / 10
print(total_number)

aggregated_line = np.mean(CD_list, axis=0) / total_number + np.mean(DC_list, axis=0) / total_number
# plt.plot(x, np.mean(CC_list, axis=0) / total_number, label='CC')
# # plt.plot(x, np.mean(CD_list, axis=0) / total_number, label='CD')
# # plt.plot(x, np.mean(DC_list, axis=0) / total_number, label='DC')
# plt.plot(x, aggregated_line, label='CD/DC')
# plt.plot(x, np.mean(DD_list, axis=0) / total_number, label='DD')

# make it smooth
print(x.shape)
f = interp1d(x, np.mean(CC_list, axis=0) / total_number, kind='cubic')
x_smooth = np.linspace(1, 20, 100)
y_smooth = f(x_smooth)
plt.plot(x_smooth, y_smooth, label='CC')
f = interp1d(x, aggregated_line, kind='cubic')
y_smooth = f(x_smooth)
plt.plot(x_smooth, y_smooth, label='CD/DC')
f = interp1d(x, np.mean(DD_list, axis=0) / total_number, kind='cubic')
y_smooth = f(x_smooth)
plt.plot(x_smooth, y_smooth, label='DD')

plt.ylim(-0.1, 1.1)
plt.legend()
plt.xlabel('Step', fontsize=12)  
plt.ylabel('Rate of pair choices', fontsize=13)
plt.title(f'Evolution of LLMs pair choices - {scenario} scenario', fontsize=14)
# make the x axis show integer only
plt.xticks(np.arange(1, 21, 1.0))
# plt.savefig(os.path.join('../create_figures/figs/', f'pair_actions_{scenario}.png'), dpi=500)
plt.show()



