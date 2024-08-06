import numpy as np
import matplotlib.pyplot as plt

observation = np.load('whole_observation_list_record.npy')

agent_ratio = np.load('whole_agent_ratio_list_record.npy')

intervention = np.load('whole_intervention_list_record.npy')


adjacency_matrix = np.load('whole_adjacency_matrix_record.npy')

# print(observation)
# print(agent_ratio)
print(np.argwhere(np.isnan(intervention)))
# print(intervention)


print(np.shape(observation))
print(np.shape(agent_ratio))
print(np.shape(intervention))
print(np.shape(adjacency_matrix))


