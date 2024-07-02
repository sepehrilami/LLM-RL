import os
import shutil
import random
import numpy as np

from agent import Agent
from utils.utils import *
from utils.provider import Provider
from utils.data_handler import DataHandler
from utils.logger import Logger


class Orbit():
    def __init__(self, orbit_settings, output_path='output'):
        self.output_path = output_path
        #
        # # Create output directory
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.makedirs(self.output_path)
        
        # Initialize logger and data handler
        self.logger = Logger(path=self.output_path)
        # self.data_handler = DataHandler(path=self.output_path)

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
        self.seed = np.random.randint(1000)
        # print(self.seed)
        random.seed(self.seed)        
        np.random.seed(self.seed)

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

        actions_temp, action_con, action_tar = agents[your_index].step(observation[your_index, opponent_index], your_index, connection_status, others_last_action_list)


        if actions_temp == 'defection':
            actions_cal = 0
        elif actions_temp == 'cooperation':
            actions_cal = 1
        else:
            print("error")
            exit()


        return actions_cal, action_con, action_tar

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
        # print(self.variables)
        
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

        # print(self.family_setting)

        # Initialize agents
        self.agent = Agent(
                orbit=self.orbit,
                agent_name=self.agent_name,
                variables = self.variables,
                actions = self.actions,
                system_prompt_template=self.family_setting['system_prompt_template'],
                prompt_template=self.family_setting['prompt_template'],
                memory_length=self.family_setting['memory_length']
            )
        return self.agent

    # def step(self):
    #     self.orbit_step += 1
    #     self.logger.info(f"Orbit step {self.orbit_step} started.")
    #     # np.random.shuffle(self.agents)
    #     if self.update_type == 'asynchronous':
    #         for agent in self.agents:
    #             agent.step()
    #             self.data_handler.collect(
    #                 orbit_step=self.orbit_step,
    #                 variables=self.variables,
    #                 actions=self.actions,
    #                 family_name=agent.family_name,
    #                 agent_name=agent.agent_name,
    #                 llm_actions=agent.llm_actions
    #             )
    #             self.logger.agent_step(
    #                 orbit_step=self.orbit_step,
    #                 family_name=agent.family_name,
    #                 agent_name=agent.agent_name,
    #                 system_prompt=agent.system_prompt,
    #                 prompt=agent.prompt,
    #                 llm_actions=agent.llm_actions
    #             )
    #             agent.update_actions()
    #             agent.update_variables()
    #
    #     elif self.update_type == 'synchronous':
    #         for agent in self.agents:
    #             agent.step()
    #             self.data_handler.collect(
    #                 orbit_step=self.orbit_step,
    #                 variables=self.variables,
    #                 actions=self.actions,
    #                 family_name=agent.family_name,
    #                 agent_name=agent.agent_name,
    #                 llm_actions=agent.llm_actions
    #             )
    #             self.logger.agent_step(
    #                 orbit_step=self.orbit_step,
    #                 family_name=agent.family_name,
    #                 agent_name=agent.agent_name,
    #                 system_prompt=agent.system_prompt,
    #                 prompt=agent.prompt,
    #                 llm_actions=agent.llm_actions
    #             )
    #         for agent in self.agents:
    #             agent.update_actions()
    #         for agent in self.agents:
    #             agent.update_variables()






            # orbit.data_handler.collect(
            #     orbit_step=orbit.orbit_step,
            #     variables=orbit.variables,
            #     actions=orbit.actions,
            #     family_name=agents[i].family_name,
            #     agent_name=agents[i].agent_name,
            #     llm_actions=agents[i].llm_actions
            # )
            # orbit.logger.agent_step(
            #     orbit_step=orbit.orbit_step,
            #     family_name=agents[i].family_name,
            #     agent_name=agents[i].agent_name,
            #     system_prompt=agents[i].system_prompt,
            #     prompt=agents[i].prompt,
            #     llm_actions=agents[i].llm_actions
            # )
        # for agent in agents:
        #     agent.update_actions()
        # for agent in agents:
        #     agent.update_variables()

