import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pair_actions = np.load('last_action/round_wise_data/rounds_total_pair_actions_20_20_10_0.25_nun2.npy', allow_pickle=True)

# data = pd.DataFrame(pair_actions)
# data = pd.DataFrame.from_dict(pair_actions)
# data = pd.DataFrame.from_records(pair_actions)
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

print(np.shape(CC_list))

x = np.arange(20)

total_number = np.sum(CC_list + CD_list + DC_list + DD_list, axis=0) / 10
print(total_number)
plt.plot(x, np.mean(CC_list, axis=0) / total_number, label='CC')
plt.plot(x, np.mean(CD_list, axis=0) / total_number, label='CD')
plt.plot(x, np.mean(DC_list, axis=0) / total_number, label='DC')
plt.plot(x, np.mean(DD_list, axis=0) / total_number, label='DD')
plt.ylim(0, 1)
plt.legend()
plt.show()



