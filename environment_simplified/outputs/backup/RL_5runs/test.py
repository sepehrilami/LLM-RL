import numpy as np

intervention = np.load('agent_intervention_matrix_10_5_1_0.25.npy')

print(np.shape(intervention))
print(intervention[0, 0])

print(np.sum(intervention is None))


# print(np.argwhere(intervention is np.nan))

# print(np.where(intervention == 2))