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
    def __init__(self, orbit_settings, family_settings, output_path='output'):
        self.output_path = output_path

        # Create output directory
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.makedirs(self.output_path)
        
        # Initialize logger and data handler
        self.logger = Logger(path=self.output_path)
        self.data_handler = DataHandler(path=self.output_path)

        self.orbit_settings = orbit_settings
        self.family_settings = family_settings

        self.orbit_step = 0

        # Unpack orbit settings
        self.orbit_name = self.orbit_settings['orbit_name']
        self.orbit_steps = self.orbit_settings['orbit_steps']
        self.step_unit = self.orbit_settings['step_unit']
        self.update_type = self.orbit_settings['update_type']
        self.seed = self.orbit_settings['seed']

        self.provider = self.orbit_settings['provider']
        self.model = self.orbit_settings['model']
        self.temperature = self.orbit_settings['temperature']
        self.api_key = self.orbit_settings['api_key']

        # Setup seed for randomness
        random.seed(self.seed)        
        np.random.seed(self.seed)

        # Initialize provider
        self.provider = Provider(
            provider=self.provider, 
            model=self.model, 
            temperature=self.temperature, 
            api_key=self.api_key,
            logger=self.logger
        )
        self.logger.info(f"Provider '{self.provider.provider}' initialized.")
        
        # Initialize agents
        # self.initialize_agents()
        # self.logger.info(f"Agents initialized.")

    def create_agents(self):
        # Global variables
        self.global_variables = {
            "step_unit": self.step_unit
        }

        # Initialize variables
        self.variables = {
            family_data['family_name']: {
                variable_data['variable_name']: {
                    f"agent_{i}": {
                        'memory': [],
                        'value': get_variable_initial_value(variable_data),
                        'update_rule': variable_data['update_rule']
                    }
                    for i in range(family_data['number_of_agents'])
                }
                for variable_data in family_data['variables']
            }
            for family_data in self.family_settings
        }
        
        # Initialize actions
        self.actions = {
            family_data['family_name']: {
                action_data['action_name']: {
                    f"agent_{i}": {
                        'memory': [],
                        'value': None,
                        'type': action_data['action_type'],
                        'description': action_data['action_description'],
                        **({'options': action_data['options']} if action_data['action_type'] == 'option' else {})
                    }
                    for i in range(family_data['number_of_agents'])
                }
                for action_data in family_data['actions']
            }
            for family_data in self.family_settings
        }

        # Initialize agents
        self.agents = [
            Agent(
                orbit=self,
                agent_name=f"agent_{i}",
                family_name=family_data['family_name'],
                system_prompt_template=family_data['system_prompt_template'],
                prompt_template=family_data['prompt_template'],
                memory_length=family_data['memory_length']
            )
            for family_data in self.family_settings
            for i in range(family_data['number_of_agents'])
        ]
        return self.agents

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

def env_step(agents, your_index, opponent_index, observation):
    # orbit.orbit_step += 1
    # orbit.logger.info(f"Orbit step {orbit.orbit_step} started.")
    # np.random.shuffle(self.agents)
    # print(np.shape(observation))
    actions_temp, reasoning_temp = agents[your_index].step(observation[your_index, opponent_index])

    if actions_temp == 'defection':
        actions_temp = 0
    elif actions_temp == 'cooperation':
        actions_temp = 1
    else:
        print("error")
        exit()

    return actions_temp, reasoning_temp





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

