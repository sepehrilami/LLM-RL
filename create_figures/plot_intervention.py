import numpy as np

import matplotlib.pyplot as plt

ratio_int = np.load('ratio_intervention/agent_intervention_matrix_20_20_10_0.25.npy')

print(np.sum(np.argwhere(ratio_int == [0,2])))

print(np.shape(ratio_int))


