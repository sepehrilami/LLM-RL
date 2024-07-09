import json
import time

import matplotlib.pyplot as plt

from orbit import Orbit, agent_individual, change_network
from datetime import datetime
import networkx as nx
import numpy as np

start_time = datetime.now()
num_agent = 5
game_length = 3


# Load settings
family_settings = json.load(open('settings/family_settings.json'))
orbit_settings = json.load(open('settings/orbit_settings.json'))

orbit = Orbit(orbit_settings)
llm_agents = [agent_individual(orbit, 'agent_test'+str(i), family_settings, 0).agent_initialization()
              for i in range(num_agent)]

observation_list = -np.ones((len(llm_agents), len(llm_agents), 2))
agent_index = np.arange(5)

# g = nx.complete_graph(n=len(llm_agents))
g = nx.erdos_renyi_graph(n=len(llm_agents), p=0.5, directed=True)

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

fig = save_graph(g, pos, step=-1)
for step in range(game_length):
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
            action_ij, action_con, action_dis = orbit.env_step(llm_agents, i, neighbor_index[j], observation_list, adj_matrix[i], observation_list[i])
            # action_ji, _, _ = orbit.env_step(llm_agents, neighbor_index[j], i, observation_list, adj_matrix[neighbor_index[j]], observation_list[neighbor_index[j]])

            observation_list[i, neighbor_index[j], 0] = action_ij
            # observation_list[i, neighbor_index[j], 1] = action_ji
            print(f'Round {step}, agent {i} vs agent {neighbor_index[j]}, action: {action_ij}, connection: {action_con}, disconnection: {action_dis}')

        change_network(g, i, action_con, action_dis)
            # print('Round ', step, ' you are agent ', i, 'opponent is agent ', neighbor_index[j],
            #       'your action is: ', action_ij, 'your connection action is to ', action_con,
            #       'your dis-connection action is to ', action_dis)
            # nx.draw_networkx(g, pos=nx.spring_layout(g))
        # print(f'Inner loop time: {time.time() - initial_time}')
    # print(f'Outer loop time: {time.time() - initial_time}')
    
    fig = save_graph(g, pos, step)
    
plt.close(fig)
print(f'Total time: {round(time.time() - initial_time, 2)}')
            # plt.show()
