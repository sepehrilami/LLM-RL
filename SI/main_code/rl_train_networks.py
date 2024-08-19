import json
import time

from datetime import datetime

from agent import LLMAgent, A2C_manager
from utils.utils import *
from langchain_core.prompts import PromptTemplate
import networkx as nx

def build_prompt(intervention, last_action1="D", last_action2="D", own_ratio="rarely",  other_ratio="rarely", neighbor_ratio="rarely"):
    if intervention == 0:
        prompt = multiple_input_prompt_mark.format(your_action=last_action1, other_action=last_action2)
    elif intervention == 1:
        prompt = multiple_input_prompt_both.format(your_action=last_action1, other_action=last_action2, your_ratio=own_ratio, other_ratio=other_ratio)
    elif intervention == 2:
        prompt = multiple_input_prompt_neighbor.format(your_action=last_action1, other_action=last_action2, your_ratio=own_ratio, neighbor_ratio=neighbor_ratio)
    elif intervention is None:
        prompt = prompts_file['template_no_history']

    return prompt

def update_C_ratio_list(C_ratio_list, index1, index2, action_1_str, action_2_str):
    if action_1_str == "C":
        new_ratio_1 = (C_ratio_list[index1, 0] * C_ratio_list[index1, 1] + 1) / (C_ratio_list[index1, 1] + 1)
    else:
        new_ratio_1 = (C_ratio_list[index1, 0] * C_ratio_list[index1, 1]) / (C_ratio_list[index1, 1] + 1)

    if action_2_str == "C":
        new_ratio_2 = (C_ratio_list[index2, 0] * C_ratio_list[index2, 1] + 1) / (C_ratio_list[index2, 1] + 1)
    else:
        new_ratio_2 = (C_ratio_list[index2, 0] * C_ratio_list[index2, 1]) / (C_ratio_list[index2, 1] + 1)

    C_ratio_list[index1, 0] = new_ratio_1
    C_ratio_list[index1, 1] += 1

    C_ratio_list[index2, 0] = new_ratio_2
    C_ratio_list[index2, 1] += 1

    return C_ratio_list

def cal_neighbor_C(g, C_ratio_list, index_own):
    neighbor_list = np.array(list(g.neighbors(index_own)))

    ratio_list = C_ratio_list[neighbor_list, 0]
    total_number_list = C_ratio_list[neighbor_list, 1]
    number_C = np.sum(np.multiply(ratio_list, total_number_list))
    total_number = np.sum(total_number_list)

    return number_C / total_number

def one_hot_last_action(observation):
    if observation == -1: # no previous history against this guy
        return [1, 0, 0]
    elif observation == 0: # defection in the last action
        return [0, 1, 0]
    elif observation == 1: # cooperation in the last action
        return [0, 0, 1]

start_time = datetime.now()
num_agent = 20
game_length = 20
training_episode = 50
# Load settings
prompts_file = json.load(open('settings/prompts.json'))
env_settings = json.load(open('settings/env_settings.json'))

a2c_manager = A2C_manager(state_dim=15, action_dim=3)
llm_agent_list = [LLMAgent(env_settings, 'llm_agent'+str(i)) for i in range(num_agent)]

initial_time = time.time()
# outer loop for rounds

prompt_option_no_history_template = prompts_file['template_no_history']
prompt_option_markovian_history_template = prompts_file['template_markovian_history']
prompt_option_both_history_template = prompts_file['template_markovian_both_ratio_history']
prompt_option_neighbor_history_template = prompts_file['template_markovian_network_ratio_history']

multiple_input_prompt_mark = PromptTemplate(
    input_variables=['your_action', 'other_action'],
    template=prompt_option_markovian_history_template
)
multiple_input_prompt_both = PromptTemplate(
    input_variables=['your_action', 'other_action', 'your_ratio', 'other_ratio'],
    template=prompt_option_both_history_template
)
multiple_input_prompt_neighbor = PromptTemplate(
    input_variables=['your_action', 'other_action', 'your_ratio', 'neighbor_ratio'],
    template=prompt_option_neighbor_history_template
)

whole_C_ratio_list = []
avg_C_ratio_list = []
for episode in range(training_episode):
    done = False

    g = nx.erdos_renyi_graph(num_agent, p=0.25)

    observation_list = -np.ones((num_agent, num_agent, 2))

    link_list = list(g.edges())
    C_ratio_list = np.zeros((num_agent, 2)) # the first value in second dim is the C_ratio, the second value is the number of actions.
    for step in range(game_length):
        # print(step)
        for (index1, index2) in link_list:
            if step < 2: # no RL intervention in first 2 steps because we always get DD in the first step with "no history" prompt
                prompt_1 = build_prompt(intervention=None)
                prompt_2 = build_prompt(intervention=None)
                action_1_str = llm_agent_list[index1].answer(prompt_1)
                action_2_str = llm_agent_list[index2].answer(prompt_2)

                last_action1_str = action_1_str
                last_action2_str = action_2_str

            else:
                if step == game_length - 1:
                    done = True

                own_frequency1 = frequence_number_to_index(C_ratio_list[index1, 0])
                own_frequency2 = frequence_number_to_index(C_ratio_list[index2, 0])

                neighbors_frequency1 = frequence_number_to_index(cal_neighbor_C(g, C_ratio_list, index1))
                neighbors_frequency2 = frequence_number_to_index(cal_neighbor_C(g, C_ratio_list, index2))

                observation1 = one_hot_last_action(observation_list[index1, index2, 0])
                observation2 = one_hot_last_action(observation_list[index1, index2, 1])

                obs_input1 = observation1 + observation2 + own_frequency1 + own_frequency2 + neighbors_frequency1
                obs_input2 = observation2 + observation1 + own_frequency2 + own_frequency1 + neighbors_frequency2

                # intervention from RL agent
                intervention1 = a2c_manager.choose_action(obs_input1)
                intervention2 = a2c_manager.choose_action(obs_input2)

                prompt_1 = build_prompt(intervention1, convert_int_to_str(observation_list[index1, index2, 0]),
                                        convert_int_to_str(observation_list[index1, index2, 1]), frequence_index_to_str(own_frequency1), frequence_index_to_str(own_frequency2), frequence_index_to_str(neighbors_frequency1))
                prompt_2 = build_prompt(intervention2, convert_int_to_str(observation_list[index1, index2, 1]),
                                        convert_int_to_str(observation_list[index1, index2, 0]), frequence_index_to_str(own_frequency2), frequence_index_to_str(own_frequency1), frequence_index_to_str(neighbors_frequency2))

                # response from LLM agent
                action_1_str = llm_agent_list[index1].answer(prompt_1)
                action_2_str = llm_agent_list[index2].answer(prompt_2)


                reward1 = get_reward(action_1_str, action_2_str)
                reward2 = get_reward(action_2_str, action_1_str)

                next_observation1 = convert_str_to_int(action_1_str)
                next_observation2 = convert_str_to_int(action_2_str)

                n_own_frequency1 = frequence_number_to_index(C_ratio_list[index1, 0])
                n_own_frequency2 = frequence_number_to_index(C_ratio_list[index2, 0])

                n_neighbors_frequency1 = frequence_number_to_index(cal_neighbor_C(g, C_ratio_list, index1))
                n_neighbors_frequency2 = frequence_number_to_index(cal_neighbor_C(g, C_ratio_list, index2))

                n_obs_input1 = one_hot_last_action(next_observation1) + one_hot_last_action(next_observation2) + n_own_frequency1 + n_own_frequency2 + n_neighbors_frequency1
                n_obs_input2 = one_hot_last_action(next_observation2) + one_hot_last_action(next_observation1) + n_own_frequency2 + n_own_frequency1 + n_neighbors_frequency2

                a2c_manager.train(obs_input1, intervention1, reward1, n_obs_input1, done)
                a2c_manager.train(obs_input2, intervention2, reward2, n_obs_input2, done)

            C_ratio_list = update_C_ratio_list(C_ratio_list, index1, index2, action_1_str, action_2_str)
            # updating last action matrix
            observation_list[index1, index2] = [convert_str_to_int(action_1_str), convert_str_to_int(action_2_str)]
            observation_list[index2, index1] = [convert_str_to_int(action_2_str), convert_str_to_int(action_1_str)]

    whole_C_ratio_list.append(C_ratio_list)
    ratio_list = C_ratio_list[:, 0]
    total_number_list = C_ratio_list[:, 1]
    number_C = np.sum(np.multiply(ratio_list, total_number_list))
    total_number = np.sum(total_number_list)
    avg_C_ratio_list.append(number_C/total_number)

    if (episode+1) % 5 == 0:
        a2c_manager.save_model(f'save_model/manager_networks_{num_agent}nodes_{episode+1}')

    print(f'Duration: {round(time.time() - initial_time, 2)}, Round:{episode}, '
          f'C ratio: {number_C/total_number}, avg C ratio: {np.mean(avg_C_ratio_list)}')

