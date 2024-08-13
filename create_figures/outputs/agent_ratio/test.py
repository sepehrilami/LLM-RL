import numpy as np

intervention = np.load('agent_intervention_matrix_20_20_10_0.25.npy')

print(np.shape(intervention))

print(np.where(intervention ==2))