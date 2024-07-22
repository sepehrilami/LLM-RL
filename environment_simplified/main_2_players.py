import json
import time

from datetime import datetime
import numpy as np
from agent import LLMAgent

def convert_str_to_int(action):
    if action == "C":
        return 0
    elif action == "D":
        return 1
    else:
        print(f'Invalid action: {action}')
        return -1

def stat_analysis(all_actions):
    # this list is a list of C and D actions.
    # count them and print the ration of C and D.
    np_actions = np.array(all_actions)
    print(f'Total actions: {len(all_actions)}')
    print(f'Total C: {np.sum(np_actions == "C")}')
    print(f'Total D: {np.sum(np_actions == "D")}')
    print(f'Ratio of C: %{round(np.sum(np_actions == "C") / len(all_actions), 2) * 100}')
    print(f'Ratio of D: %{round(np.sum(np_actions == "D") / len(all_actions), 2) * 100}')
    print("--------------------")


start_time = datetime.now()
num_agent = 2
game_length = 100

# Load settings
prompts_file = json.load(open('settings/prompts.json'))
env_settings = json.load(open('settings/env_settings.json'))

# print(prompts_file['prompt_no_history'])

llm_agent_list = [LLMAgent(env_settings, 'llm_agent'+str(i)) for i in range(num_agent)]

initial_time = time.time()
# outer loop for rounds
all_actions1 = []
all_actions2 = []
for step in range(game_length):
    # print(step)
    prompt_1 = prompts_file['prompt_no_history']
    prompt_2 = prompts_file['prompt_no_history']

    action_1 = llm_agent_list[0].answer(prompt_1)
    action_2 = llm_agent_list[1].answer(prompt_2)
    # print(action_1, action_2)
    all_actions1.append(action_1)
    all_actions2.append(action_2)
    # inner loop for agents
stat_analysis(all_actions1)
stat_analysis(all_actions2)

print(f'Total time: {round(time.time() - initial_time, 2)}')

