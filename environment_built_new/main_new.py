import json
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

observation_list = np.zeros((len(llm_agents), len(llm_agents), 2))
g = nx.complete_graph(n=len(llm_agents))
game_length = 3
'''
Change the api keys before you run this in 'settings\orbit_settings.json'.

This is a simple test of the fundamental components, llm agents and network connection. 
the current simple environment contains several agents with a complete network structure, 
they are playing PD with others with connections, llm agents will base on the last pair of 
actions to make decisions in the current round. 
'''

# print(llm_agents)

for step in range(game_length):
    for i in range(len(llm_agents)):
        llm_agents[i].orbit.orbit_step += 1
        neighbor_index = list(g.neighbors(i))
        # print(neighbor_index)
        for j in range(len(neighbor_index)):
            # print(neighbor_index[j])
            action_ij, reasoning_ij = orbit.env_step(llm_agents, i, neighbor_index[j], observation_list)
            observation_list[i, neighbor_index[j]] = action_ij
            print('Round ', step, ' you are agent ', i, 'opponent is agent ', neighbor_index[j],
                  'your action is: ', action_ij, 'your reasoning is: ', reasoning_ij)

