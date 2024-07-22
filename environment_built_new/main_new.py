import json
import time

import matplotlib.pyplot as plt
import torch
from orbit import Orbit, agent_individual, change_network
from datetime import datetime
import networkx as nx
import numpy as np

# from numba import cuda
# device = cuda.get_current_device()


start_time = datetime.now()
num_agent = 10
game_length = 1


# Load settings
family_settings = json.load(open('settings/family_settings.json'))
orbit_settings = json.load(open('settings/orbit_settings.json'))

orbit = Orbit(orbit_settings)
llm_agents = [agent_individual(orbit, 'agent_test'+str(i), family_settings, 0).agent_initialization()
              for i in range(num_agent)]

observation_list = -np.ones((len(llm_agents), len(llm_agents), 2))
agent_index = np.arange(5)

# g = nx.complete_graph(n=len(llm_agents))
g = nx.erdos_renyi_graph(n=len(llm_agents), p=1, directed=False)

pos = nx.kamada_kawai_layout(g)

'''
Change the api keys before you run this in 'settings\orbit_settings.json'.

This is a simple test of the fundamental components, llm agents and network connection. 
the current simple environment contains several agents with a complete network structure, 
they are playing PD with others with connections, llm agents will base on the last pair of 
actions to make decisions in the current round. 
'''


initial_time = time.time()
# outer loop for rounds
def save_graph(g, pos, step):
    plt.clf()
    fig = plt.figure()
    # pos = nx.kamada_kawai_layout(g)

    nx.draw_networkx(g, pos, ax=fig.add_subplot())
    fig.savefig(f'figure/step_{step}.png')
    return fig

def convert_str_to_int(action):
    if action == "C":
        return 0
    elif action == "D":
        return 1
    else:
        print(f'Invalid action: {action}')
        return -1
    
def stat_analysis(all_actuons):
    # this list is a list of C and D actions.
    # count them and print the ration of C and D.
    np_actions = np.array(all_actions)
    print(f'Total actions: {len(all_actions)}')
    print(f'Total C: {np.sum(np_actions == "C")}')
    print(f'Total D: {np.sum(np_actions == "D")}')
    print(f'Ratio of C: %{round(np.sum(np_actions == "C") / len(all_actions), 2) * 100}')
    print(f'Ratio of D: %{round(np.sum(np_actions == "D") / len(all_actions), 2) * 100}')
    print("--------------------")

fig = save_graph(g, pos, step=-1)
for step in range(game_length):
    all_actions = []
    # inner loop for agents
    for i in range(len(llm_agents)):
        llm_agents[i].orbit.orbit_step += 1
        neighbor_index = list(g.neighbors(i))
        adj_matrix = nx.adjacency_matrix(g).todense()
        # print(neighbor_index)
        # inner loop for neighbors of the agent

        action_con = -1
        action_dis = -1

        for j in range(len(neighbor_index)):
            # print(neighbor_index[j])
            # device.reset()
            # torch.cuda.empty_cache()
            flag = True
            time_try = 0

            action_ij = orbit.env_step(llm_agents, i, neighbor_index[j], observation_list, adj_matrix[i], observation_list[i])
            all_actions.append(action_ij)
            # while flag:
                
            #     try:
            #         # action_ij, action_con, action_dis = orbit.env_step(llm_agents, i, neighbor_index[j], observation_list, adj_matrix[i], observation_list[i])
            #         action_ij = orbit.env_step(llm_agents, i, neighbor_index[j], observation_list, adj_matrix[i], observation_list[i])
                    
            #         flag = False
            #     except:
            #         time_try += 1
            #         print(f'error of outputs from LLM, try again, {time_try} attempts')
            #         if time_try > 10:
            #             print(f'exceed the maximum of trials, break the loop')
            #             break
            # # action_ji, _, _ = orbit.env_step(llm_agents, neighbor_index[j], i, observation_list, adj_matrix[neighbor_index[j]], observation_list[neighbor_index[j]])

            observation_list[i, neighbor_index[j], 0] = convert_str_to_int(action_ij)
            # observation_list[i, neighbor_index[j], 1] = action_ji
            # print(f'Round {step}, agent {i} vs agent {neighbor_index[j]}, action: {action_ij}')

        # change_network(g, i, action_con, action_dis)
    
    # fig = save_graph(g, pos, step)
    stat_analysis(all_actions)
    
# plt.close(fig)
print(f'Total time: {round(time.time() - initial_time, 2)}')

# plt.show()
