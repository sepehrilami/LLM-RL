import numpy as np

intervention = np.load('agent_intervention_matrix_20_20_2_0.25.npy')

print(np.shape(intervention))
print(np.argwhere(intervention==2))

# print(np.where(intervention == 2))