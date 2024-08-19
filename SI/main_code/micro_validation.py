import json
import time

from datetime import datetime
from agent import LLMAgent
from utils.utils import *

start_time = datetime.now()
num_agent = 1
game_length = 100

# Load settings
prompts_file = json.load(open('settings/prompts.json'))
env_settings = json.load(open('settings/env_settings.json'))

llm_agent_list = [LLMAgent(env_settings, 'llm_agent'+str(i)) for i in range(num_agent)]

initial_time = time.time()
# outer loop for rounds

all_actions1 = []

print('Start playing...')
all_keys = ['template_no_history']
prompt_list = [prompts_file[key] for key in all_keys]
for i, prompt_1 in enumerate(prompt_list):
    all_actions1 = []
    filename = f'all_actions_{all_keys[i]}'
    print(f'Starting prompt: {all_keys[i]}')
    for step in range(game_length):

        action_1 = llm_agent_list[0].answer(prompt_1)
        all_actions1.append(action_1)

        with open(f'micro-val-results/{filename}.json', 'w') as f:
                json.dump(all_actions1, f)

        # if step % 10 == 0:
        #     time.sleep(5)

    stat_analysis(all_actions1)
print(f'Total time: {round(time.time() - initial_time, 2)}')

