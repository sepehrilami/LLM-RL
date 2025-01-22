import json
import time

from datetime import datetime
from agent import LLMAgent
from utils.utils import *
from langchain_core.prompts import PromptTemplate

start_time = datetime.now()
num_agent = 20
game_length = 200

# Load settings
prompts_file = json.load(open('settings/prompts.json'))
env_settings = json.load(open('settings/env_settings.json'))
# prompt_option_no_history_template = prompts_file['prompt_no_history']
# prompt_option_markovian_history_template = prompts_file['prompt_markovian_history']

# multiple_input_prompt = PromptTemplate(
#     input_variables=['own_frequence', 'neighbors_frequence'],
#     template=prompt_option_no_history_template
# )


llm_agent_list = [LLMAgent(env_settings, 'llm_agent'+str(i)) for i in range(num_agent)]

initial_time = time.time()
# outer loop for rounds


all_actions1 = []
# all_actions2 = []
# freq1 = 'occasionally'
# freq2 = 'sometimes'
# print(f'Agent 1: {freq1}, Agent 2: {freq2}')
print('Start playing...')
all_keys = ['template_no_history']
prompt_list = [prompts_file[key] for key in all_keys]
for i, prompt_1 in enumerate(prompt_list):
    all_actions1 = []
    filename = f'all_actions_{all_keys[i]}'
    print(f'Starting prompt: {all_keys[i]}')
    for step in range(game_length):
        # prompt_1 = prompts_file['prompt_no_history']
        # prompt_1 = multiple_input_prompt.format(own_frequence=freq1, neighbors_frequence=freq2)
        # prompt_2 = multiple_input_prompt.format(own_frequence=freq2, neighbors_frequence=freq1)

        action_1 = llm_agent_list[0].answer(prompt_1)
        # action_2 = llm_agent_list[1].answer(prompt_2)
        all_actions1.append(action_1)

        # with open(f'micro-val-results/{filename}.json', 'w') as f:
        #         json.dump(all_actions1, f)

        # if step % 10 == 0:
        #     time.sleep(5)

    # all_actions2.append(action_2)
    stat_analysis(all_actions1)
# stat_analysis(all_actions2)

print(f'Total time: {round(time.time() - initial_time, 2)}')

