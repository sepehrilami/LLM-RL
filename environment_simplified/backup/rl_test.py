import json
import time

from datetime import datetime

import numpy as np

from agent import LLMAgent, A2C_manager
from utils.utils import *
from langchain_core.prompts import PromptTemplate

def build_prompt(intervention, last_action1, last_action2):
    if intervention == 1:
        prompt = multiple_input_prompt.format(your_action=last_action1, other_action=last_action2)
    else:
        prompt = prompts_file['prompt_no_history']

    return prompt

def intervention3(observation1, observation2):
    if observation1 == 0:
        if observation2 == 0:
            intervention = 1
        else:
            intervention = 0
    else:
        if observation2 == 0:
            intervention = 0
        else:
            intervention = 1
    return intervention


start_time = datetime.now()
num_agent = 2
game_length = 10
testing_episode = 100
# Load settings
prompts_file = json.load(open('settings/prompts.json'))
env_settings = json.load(open('settings/env_settings.json'))

a2c_manager = A2C_manager(state_dim=2, action_dim=2)
llm_agent_list = [LLMAgent(env_settings, 'llm_agent'+str(i)) for i in range(num_agent)]

a2c_manager.load_model('save_model/manager_999')

initial_time = time.time()
# outer loop for rounds

prompt_option_no_history_template = prompts_file['prompt_no_history']
prompt_option_markovian_history_template = prompts_file['prompt_markovian_history']

multiple_input_prompt = PromptTemplate(
    input_variables=['your_action', 'other_action'],
    template=prompt_option_markovian_history_template
)

C_ratio_list1 = []
C_ratio_list2 = []
observation_list = []
intervention_list = []
for episode in range(testing_episode):
    last_action1 = "C"
    last_action2 = "C"
    done = False
    all_actions1 = []
    all_actions2 = []

    for step in range(game_length):
        # print(step)

        if step == 0:
            prompt_1 = build_prompt(0, last_action1, last_action2)
            prompt_2 = build_prompt(0, last_action2, last_action1)
            action_1_str = llm_agent_list[0].answer(prompt_1)
            action_2_str = llm_agent_list[1].answer(prompt_2)
            # action_1_str = "D"
            # action_2_str = "D"
            # print(action_1_str, action_2_str)
            last_action1_str = action_1_str
            last_action2_str = action_2_str
            all_actions1.append(action_1_str)
            all_actions2.append(action_2_str)

            observation1 = convert_str_to_int(last_action1_str)
            observation2 = convert_str_to_int(last_action2_str)

        else:
            if step == game_length-1:
                done = True

            # intervention1 = a2c_manager.choose_action([observation1, observation2])
            # intervention2 = a2c_manager.choose_action([observation2, observation1])
            # observation_list.append([observation1, observation2])
            # observation_list.append([observation2, observation1])
            # intervention_list.append(intervention1)
            # intervention_list.append(intervention2)


            # baseline
            # intervention1 = np.random.choice(2)
            # intervention2 = np.random.choice(2)
            # intervention1 = 1
            # intervention2 = 1
            intervention1 = intervention3(observation1, observation2)
            intervention2 = intervention3(observation2, observation1)

            # print(last_action1_str, last_action2_str)

            prompt_1 = build_prompt(intervention1, convert_int_to_str(observation1), convert_int_to_str(observation2))
            prompt_2 = build_prompt(intervention2, convert_int_to_str(observation2), convert_int_to_str(observation1))

            action_1_str = llm_agent_list[0].answer(prompt_1)
            action_2_str = llm_agent_list[1].answer(prompt_2)

            all_actions1.append(action_1_str)
            all_actions2.append(action_2_str)

            reward1 = get_reward(action_1_str)
            reward2 = get_reward(action_2_str)

            next_observation1 = convert_str_to_int(action_1_str)
            next_observation2 = convert_str_to_int(action_2_str)

            observation1 = next_observation1
            observation2 = next_observation2

    C_ratio1 = get_C_ratio(all_actions1)
    C_ratio2 = get_C_ratio(all_actions2)
    C_ratio_list1.append(C_ratio1)
    C_ratio_list2.append(C_ratio2)

    print(f'Duration: {round(time.time() - initial_time, 2)}, Round:{episode}, C ratio 1: {C_ratio1}, C ratio 2: {C_ratio2}, '
          f'avg C ratio 1: {np.mean(C_ratio_list1)}, avg C ratio 2: {np.mean(C_ratio_list2)}')

# np.save('outputs/observation_rl', np.array(observation_list))
# np.save('outputs/intervention_rl', np.array(intervention_list))