import json

import matplotlib.pyplot as plt

from orbit import Orbit, agent_individual
from datetime import datetime
import networkx as nx
import numpy as np

num_agent = 5

start_time = datetime.now()
# Load settings
family_settings = json.load(open('settings/family_settings.json'))
orbit_settings = json.load(open('settings/orbit_settings.json'))

orbit = Orbit(orbit_settings)
llm_agents = [agent_individual(orbit, 'agent_test'+str(i), family_settings, 0).agent_initialization()
              for i in range(num_agent)]

observation_list = -np.ones((len(llm_agents), len(llm_agents), 2))
agent_index = np.arange(5)
g = nx.complete_graph(n=len(llm_agents))

# g = nx.erdos_renyi_graph(n=len(llm_agents), p=0.5, directed=True)
# print(list(g.neighbors(0)))
adj_matrix = nx.adjacency_matrix(g).todense()
# nx.draw_networkx(g, pos=nx.spring_layout(g))
# plt.show()

game_length = 3

def change_network(g, agent_index, action_con, action_tar):

    if action_con == 'keep':
        return g
    elif action_con == 'connect':
        g.add_edge(agent_index, int(action_tar))
    elif action_con == 'disconnect':
        try:
            g.remove_edge(agent_index, int(action_tar))
        except:
            return g
    else:
        print('error: connection action is not available')
        exit()

    return g



'''
Change the api keys before you run this in 'settings\orbit_settings.json'.

This is a simple test of the fundamental components, llm agents and network connection. 
the current simple environment contains several agents with a complete network structure, 
they are playing PD with others with connections, llm agents will base on the last pair of 
actions to make decisions in the current round. 
'''

# print(llm_agents)
pos = nx.kamada_kawai_layout(g)
for step in range(game_length):
    for i in range(len(llm_agents)):
        llm_agents[i].orbit.orbit_step += 1
        neighbor_index = list(g.neighbors(i))
        adj_matrix = nx.adjacency_matrix(g).todense()
        # print(neighbor_index)
        for j in range(len(neighbor_index)):
            # print(neighbor_index[j])
            action_ij, action_con, action_tar = orbit.env_step(llm_agents, i, neighbor_index[j], observation_list, adj_matrix[i], observation_list[i])
            action_ji, _, _ = orbit.env_step(llm_agents, neighbor_index[j], i, observation_list, adj_matrix[neighbor_index[j]], observation_list[neighbor_index[j]])

            observation_list[i, neighbor_index[j], 0] = action_ij
            observation_list[i, neighbor_index[j], 1] = action_ji

            g = change_network(g, i, action_con, action_tar)
            print('Round ', step, ' you are agent ', i, 'opponent is agent ', neighbor_index[j],
                  'your action is: ', action_ij, 'your connection action is to ', action_con, 'your target agent is: ', action_tar)
            # nx.draw_networkx(g, pos=nx.spring_layout(g))
            plt.clf()
            fig = plt.figure()
            nx.draw_networkx(g, pos, ax=fig.add_subplot())
            fig.savefig('figure/step_' + str(step) + '_' + str(i) + '_'+ str(neighbor_index[j])+ '.png')
            plt.close(fig)
            # plt.show()
