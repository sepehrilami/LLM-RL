import numpy as np

intervention = np.load('agent_intervention_matrix_10_2_2_0.25_qil1.npy')

print(np.shape(intervention))

print(np.sum(intervention==0))


# print(np.argwhere(intervention is np.nan))

# print(np.where(intervention == 2))