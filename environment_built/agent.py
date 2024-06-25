import json

from utils.utils import *
from utils.expression_evaluator import ExpressionEvaluator


class Agent():
    def __init__(self, orbit, agent_name, family_name, system_prompt_template, prompt_template, memory_length):
        self.expression_evaluator = ExpressionEvaluator(logger=orbit.logger)

        self.orbit = orbit
        self.agent_name = agent_name
        self.family_name = family_name

        self.system_prompt_template = system_prompt_template
        self.prompt_template = prompt_template

        self.global_variables = self.orbit.global_variables

        self.memory_length = memory_length
        self.memory = {
            'system_prompt': None,
            'prompt': [],
            'llm_actions': []
        }
            
    def step(self, observation):
        # Generate prompts

        self.orbit.variables[self.family_name]['history_you'][self.agent_name]['value'] = observation[0]
        self.orbit.variables[self.family_name]['history_other'][self.agent_name]['value'] = observation[1]

        merged_data = merge_dicts(self.orbit.variables, self.orbit.actions)
        self.system_prompt = self.expression_evaluator.evaluate(
            expression=self.system_prompt_template,
            data=merged_data,
            global_data=self.orbit.global_variables,
            family_name=self.family_name,
            agent_name=self.agent_name,
            literal=True
        )
        self.prompt = self.expression_evaluator.evaluate(
            expression=self.prompt_template,
            data=merged_data,
            global_data=self.orbit.global_variables,
            family_name=self.family_name,
            agent_name=self.agent_name,
            literal=True
        )
        # print(self.prompt)
        # Update memory
        self.memory['system_prompt'] = self.system_prompt
        self.memory['prompt'].append(self.prompt)

        # print(self.orbit.actions)
        # Query llm
        self.llm_actions = self.orbit.provider.query_llm(
            memory=self.memory, 
            memory_length=self.memory_length,
            actions=self.orbit.actions,
            family_name=self.family_name,
            agent_name=self.agent_name
        )
        # print(self.llm_actions)
        # print(self.llm_actions['money_to_contribute']['action'])
        # Update memory
        self.memory['llm_actions'].append(json.dumps(self.llm_actions))
        # print(self.llm_actions)

        return self.llm_actions['decision']['action'], self.llm_actions['decision']['reasoning']

    def update_actions(self):
        # Update actions first as they may be used in variable update rules
        for action_name, action_data in self.orbit.actions[self.family_name].items():
            new_value = self.llm_actions[action_name]['action']
            action_data[self.agent_name]["value"] = new_value
            action_data[self.agent_name]["memory"].append(new_value)

    def update_variables(self):
        # Update variables
        merged_data = merge_dicts(self.orbit.variables, self.orbit.actions)
        
        for variable_data in self.orbit.variables[self.family_name].values():
            if variable_data[self.agent_name]['update_rule']:
                variable = variable_data[self.agent_name]
                new_value = self.expression_evaluator.evaluate(
                    expression=variable['update_rule'],
                    data=merged_data,
                    global_data=self.global_variables,
                    family_name=self.family_name,
                    agent_name=self.agent_name,
                    literal=False
                )
                variable["value"] = new_value
                variable["memory"].append(variable["value"])
            else:
                variable_data[self.agent_name]["memory"].append(variable_data[self.agent_name]["value"])