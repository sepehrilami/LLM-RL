import json
import time

from datetime import datetime
from agent import LLMAgent
from utils.utils import *
from langchain_core.prompts import PromptTemplate

start_time = datetime.now()
num_agent = 2
game_length = 100

# Load settings
prompts_file = json.load(open('settings/prompts.json'))
env_settings = json.load(open('settings/env_settings.json'))
prompt_option_no_history_template = prompts_file['prompt_no_history']
prompt_option_markovian_history_template = prompts_file['prompt_markovian_history']

multiple_input_prompt = PromptTemplate(
    input_variables=['your_action', 'other_action'],
    template=prompt_option_markovian_history_template
)


llm_agent_list = [LLMAgent(env_settings, 'llm_agent'+str(i)) for i in range(num_agent)]

initial_time = time.time()
# outer loop for rounds
all_actions1 = []
all_actions2 = []
for step in range(game_length):
    # print(step)
    prompt_1 = prompts_file['prompt_no_history']
    prompt_2 = prompts_file['prompt_no_history']
    # prompt_1 = multiple_input_prompt.format(your_action="C", other_action="C")
    # prompt_2 = multiple_input_prompt.format(your_action="C", other_action="C")

    action_1 = llm_agent_list[0].answer(prompt_1)
    action_2 = llm_agent_list[1].answer(prompt_2)
    # print(action_1, action_2)
    all_actions1.append(action_1)
    all_actions2.append(action_2)
    # inner loop for agents
stat_analysis(all_actions1)
stat_analysis(all_actions2)

print(f'Total time: {round(time.time() - initial_time, 2)}')

