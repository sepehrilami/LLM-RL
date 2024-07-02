import json

from utils.utils import *
from utils.expression_evaluator import ExpressionEvaluator
from langchain_core.prompts import PromptTemplate

# langchain_core.prompts.PromptTemplate
class Agent():
    def __init__(self, orbit, agent_name, variables, actions, system_prompt_template, prompt_template, memory_length):
        self.expression_evaluator = ExpressionEvaluator(logger=orbit.logger)

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
        # Generate prompts
        # print(self.variables['history_you'])

        # self.variables['history_you']['value'] = observation[0]
        # self.variables['history_other']['value'] = observation[1]
        # self.variables['your_index']['value'] = your_index
        # self.variables['connection_status']['value'] = connection_status
        # self.variables['others_last_action_list']['value'] = others_last_action_list


        # merged_data = merge_dicts(self.variables, self.actions)

        # self.system_prompt = self.expression_evaluator.evaluate(
        #     expression=self.system_prompt_template,
        #     data=merged_data,
        #     global_data=self.orbit.global_variables,
        #     agent_name=self.agent_name,
        #     literal=True
        # )
        # self.prompt = self.expression_evaluator.evaluate(
        #     expression=self.prompt_template,
        #     data=merged_data,
        #     global_data=self.orbit.global_variables,
        #     agent_name=self.agent_name,
        #     literal=True
        # )

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

        # print(self.variables['history_you']['value'])
        self.system_prompt = self.system_prompt_template
        self.prompt = multiple_input_prompt.format(history_you=str(observation[0]),
                                                   history_other=str(observation[1]),
                                                   your_index=your_index,
                                                   connection_status=connection_status,
                                                   others_last_action_list=others_last_action_list
                                                   )
        # print(self.prompt)
        # Update memory
        self.memory['system_prompt'] = self.system_prompt
        self.memory['prompt'].append(self.prompt)

        # print(self.orbit.actions)
        # for item in self.actions.items():
        #     print(item[1]['type'])
        # Query llm
        self.llm_actions = self.orbit.provider.query_llm(
            memory=self.memory, 
            memory_length=self.memory_length,
            actions=self.actions,
            agent_name=self.agent_name
        )
        # print(self.llm_actions)
        # print(self.llm_actions['money_to_contribute']['action'])
        # Update memory
        self.memory['llm_actions'].append(json.dumps(self.llm_actions))
        # print(self.llm_actions)

        return self.llm_actions['decision']['action'], self.llm_actions['decision_con']['action'], self.llm_actions['decision_dis']['action']

    # def update_actions(self):
    #     # Update actions first as they may be used in variable update rules
    #     for action_name, action_data in self.orbit.actions[self.family_name].items():
    #         new_value = self.llm_actions[action_name]['action']
    #         action_data[self.agent_name]["value"] = new_value
    #         action_data[self.agent_name]["memory"].append(new_value)
    #
    # def update_variables(self):
    #     # Update variables
    #     merged_data = merge_dicts(self.orbit.variables, self.orbit.actions)
    #
    #     for variable_data in self.orbit.variables[self.family_name].values():
    #         if variable_data[self.agent_name]['update_rule']:
    #             variable = variable_data[self.agent_name]
    #             new_value = self.expression_evaluator.evaluate(
    #                 expression=variable['update_rule'],
    #                 data=merged_data,
    #                 global_data=self.global_variables,
    #                 family_name=self.family_name,
    #                 agent_name=self.agent_name,
    #                 literal=False
    #             )
    #             variable["value"] = new_value
    #             variable["memory"].append(variable["value"])
    #         else:
    #             variable_data[self.agent_name]["memory"].append(variable_data[self.agent_name]["value"])