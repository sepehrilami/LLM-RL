import json
from orbit import Orbit, env_step
from datetime import datetime
import networkx as nx
import numpy as np

start_time = datetime.now()
# Load settings
family_settings = json.load(open('settings/family_settings.json'))
orbit_settings = json.load(open('settings/orbit_settings.json'))

orbit = Orbit(orbit_settings, family_settings)
llm_agents = orbit.create_agents()

observation_list = np.zeros((len(llm_agents), len(llm_agents), 2))
g = nx.complete_graph(n=len(llm_agents))

'''
Change the api keys before you run this in 'settings\orbit_settings.json'.

This is a simple test of the fundamental components, llm agents and network connection. 
the current simple environment contains several agents with a complete network structure, 
they are playing PD with others with connections, llm agents will base on the last pair of 
actions to make decisions in the current round. 
'''

for step in range(orbit.orbit_steps):
    for i in range(len(llm_agents)):
        llm_agents[i].orbit.orbit_step += 1
        neighbor_index = list(g.neighbors(i))
        # print(neighbor_index)
        for j in range(len(neighbor_index)):
            # print(neighbor_index[j])
            action_ij, reasoning_ij = env_step(llm_agents, i, neighbor_index[j], observation_list)
            observation_list[i, neighbor_index[j]] = action_ij
            print('Round ', step, ' you are agent ', i, 'opponent is agent ', neighbor_index[j],
                  'your action is: ', action_ij, 'your reasoning is: ', reasoning_ij)

