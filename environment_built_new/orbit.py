import os
import shutil
import random
import numpy as np

from agent import Agent
from utils.utils import *
from utils.provider import Provider


def change_network(g, agent_index, action_con, action_dis):
    action_con = int(action_con)
    action_dis = int(action_dis)

    if action_con == -1:
        pass
    else:
        g.add_edge(agent_index, int(action_con))

    if action_dis == -1:
        pass
    else:
        try:
            g.remove_edge(agent_index, int(action_dis))
        except:
            pass

class Orbit():
    def __init__(self, orbit_settings, output_path='output'):
        self.output_path = output_path
        #
        # # Create output directory
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.makedirs(self.output_path)
        
        # Initialize logger and data handler

        self.orbit_settings = orbit_settings
        self.orbit_step = 0
        self.global_variables = self.orbit_settings['step_unit']
        # Unpack orbit settings
        self.seed = self.orbit_settings['seed']
        self.provider = self.orbit_settings['provider']
        self.model = self.orbit_settings['model']
        self.temperature = self.orbit_settings['temperature']
        self.api_key = self.orbit_settings['api_key']

        # Setup seed for randomness
        # self.seed = np.random.randint(1000)
        # random.seed(self.seed)
        # np.random.seed(self.seed)

        # Initialize provider
        self.provider = Provider(
            provider=self.provider, 
            model=self.model, 
            temperature=self.temperature, 
            api_key=self.api_key,
        )
        # Initialize agents
        # self.initialize_agents()
        # self.logger.info(f"Agents initialized.")

    def env_step(self, agents, your_index, opponent_index, observation, connection_status, others_last_action_list):
        # orbit.orbit_step += 1

        # actions_temp, action_con, action_tar = agents[your_index].step(observation[your_index, opponent_index], your_index, connection_status, others_last_action_list)
        actions_temp = agents[your_index].step(observation[your_index, opponent_index], your_index, opponent_index, connection_status, others_last_action_list)

        # return actions_temp, action_con, action_tar
        return actions_temp

class agent_individual():
    def __init__(self, orbit, agent_name, family_setting, memory_length):

        self.orbit = orbit
        self.agent_name = agent_name
        self.family_setting = family_setting
        self.system_prompt_template = family_setting['system_prompt_template']
        self.prompt_template = family_setting['prompt_template']
        self.memory_length = memory_length

        # self.agent_initialization()

    def agent_initialization(self):
        # Initialize variables
        self.variables = {
                variable_data['variable_name']:
                    {
                        'memory': [],
                        'value': get_variable_initial_value(variable_data),
                        'update_rule': variable_data['update_rule']
                    }
                for variable_data in self.family_setting['variables']
            }

        # Initialize actions
        self.actions = {
                action_data['action_name']:
                {
                        'memory': [],
                        'value': None,
                        'type': action_data['action_type'],
                        'description': action_data['action_description'],
                        **({'options': action_data['options']} if action_data['action_type'] == 'option' else {})
                }
                for action_data in self.family_setting['actions']
            }

        # Initialize agents
        self.agent = Agent(
                orbit=self.orbit,
                agent_name=self.agent_name,
                variables=self.variables,
                actions=self.actions,
                system_prompt_template=self.family_setting['system_prompt_template'],
                prompt_template=self.family_setting['prompt_template'],
                memory_length=self.family_setting['memory_length']
            )
        return self.agent
