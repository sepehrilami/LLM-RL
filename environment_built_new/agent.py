import json

from utils.utils import *
from langchain_core.prompts import PromptTemplate

# langchain_core.prompts.PromptTemplate
class Agent():
    def __init__(self, orbit, agent_name, variables, actions, system_prompt_template, prompt_template, memory_length):

        self.orbit = orbit
        self.agent_name = agent_name
        self.variables = variables
        self.actions = actions

        self.system_prompt_template = system_prompt_template
        self.prompt_template = prompt_template

        self.global_variables = self.orbit.global_variables

        self.memory_length = memory_length
        self.memory = {
            'system_prompt': None,
            'prompt': [],
            'llm_actions': []
        }
            
    def step(self, observation, your_index, connection_status, others_last_action_list):

        self.variables['history_you']['value'] = observation[0]
        self.variables['history_other']['value'] = observation[1]
        self.variables['your_index']['value'] = your_index
        self.variables['connection_status']['value'] = connection_status
        self.variables['others_last_action_list']['value'] = others_last_action_list

        multiple_input_prompt = PromptTemplate(
            input_variables=['history_you', 'history_other', 'your_index',
                             'connection_status', 'others_last_action_list'],
            template=self.prompt_template
        )

        self.system_prompt = self.system_prompt_template
        self.prompt = multiple_input_prompt.format(history_you=str(observation[0]),
                                                   history_other=str(observation[1]),
                                                   your_index=your_index,
                                                   connection_status=connection_status,
                                                   others_last_action_list=others_last_action_list
                                                   )
        # Update memory
        self.memory['system_prompt'] = self.system_prompt
        self.memory['prompt'].append(self.prompt)

        # Query llm
        self.llm_actions = self.orbit.provider.query_llm(
            memory=self.memory, 
            memory_length=self.memory_length,
            actions=self.actions,
            agent_name=self.agent_name
        )

        # Update memory
        self.memory['llm_actions'].append(json.dumps(self.llm_actions))

        return self.llm_actions['decision']['action'], self.llm_actions['decision_con']['action'], self.llm_actions['decision_dis']['action']
