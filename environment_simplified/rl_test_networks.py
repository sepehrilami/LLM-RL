import os
import json
import time

from agent import LLMAgent, A2C_manager
from utils.utils import *

import numpy as np
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
        prompt = prompts_file['prompt_no_history']
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
    
def choose_intervention(intervention_type, obs_input1, obs_input2):
    if intervention_type == 'RL':
        intervention1 = a2c_manager.choose_action(obs_input1)
        intervention2 = a2c_manager.choose_action(obs_input2)
    elif intervention_type == 'last_action':
        intervention1 = 0
        intervention2 = 0
    elif intervention_type == 'agent_ratio':
        intervention1 = 1
        intervention2 = 1
    elif intervention_type == 'network_ratio':
        intervention1 = 2
        intervention2 = 2
    elif intervention_type == 'randomized':
        intervention1 = np.random.randint(3)
        intervention2 = np.random.randint(3)
    else:
        raise ValueError(f'Intervention type {intervention_type} not recognized.')
    return intervention1, intervention2

def connected_erdos_renyi_graph(n, p):
    while True:
        g = nx.erdos_renyi_graph(n, p)
        if nx.is_connected(g):
            return g    

def create_prompt(intervention_type, g, C_ratio_list, observation_list, index1, index2, step):
    if step < 2:
        intervention1 = None
        intervention2 = None

        prompt_1 = build_prompt(intervention1)
        prompt_2 = build_prompt(intervention2)

        return prompt_1, prompt_2, intervention1, intervention2
    else:
        own_frequency1 = frequence_number_to_index(C_ratio_list[index1, 0])
        own_frequency2 = frequence_number_to_index(C_ratio_list[index2, 0])

        neighbors_frequency1 = frequence_number_to_index(cal_neighbor_C(g, C_ratio_list, index1))
        neighbors_frequency2 = frequence_number_to_index(cal_neighbor_C(g, C_ratio_list, index2))

        observation1 = one_hot_last_action(observation_list[index1, index2, 0])
        observation2 = one_hot_last_action(observation_list[index1, index2, 1])

        obs_input1 = observation1 + observation2 + own_frequency1 + own_frequency2 + neighbors_frequency1
        obs_input2 = observation2 + observation1 + own_frequency2 + own_frequency1 + neighbors_frequency2

        intervention1, intervention2 = choose_intervention(intervention_type, obs_input1, obs_input2)
            
        prompt_1 = build_prompt(intervention1, convert_int_to_str(observation_list[index1, index2, 0]),
                                convert_int_to_str(observation_list[index1, index2, 1]), frequence_index_to_str(own_frequency1), frequence_index_to_str(own_frequency2), frequence_index_to_str(neighbors_frequency1))
        prompt_2 = build_prompt(intervention2, convert_int_to_str(observation_list[index1, index2, 1]),
                                convert_int_to_str(observation_list[index1, index2, 0]), frequence_index_to_str(own_frequency2), frequence_index_to_str(own_frequency1), frequence_index_to_str(neighbors_frequency2))

    return prompt_1, prompt_2, intervention1, intervention2

def get_LLM_response(llm_agent_list, prompt_1, prompt_2):
    action_1_str = llm_agent_list[index1].answer(prompt_1)
    action_2_str = llm_agent_list[index2].answer(prompt_2)

    return action_1_str, action_2_str

def save_everything(whole_agent_ratio_list_record, whole_observation_list_record, whole_intervention_list_record, whole_adjacency_matrix_record, intervention_type, run_spec):
    dir = f'outputs/{intervention_type}'
    np.save(f'{os.path.join(dir, f"agent_ratio_matrix_{run_spec}")}', np.array(whole_agent_ratio_list_record))
    np.save(f'{os.path.join(dir, f"agent_last_action_matrix_{run_spec}")}', np.array(whole_observation_list_record))
    np.save(f'{os.path.join(dir, f"agent_intervention_matrix_{run_spec}")}', np.array(whole_intervention_list_record))
    np.save(f'{os.path.join(dir, f"agent_adjacency_{run_spec}")}', np.array(whole_adjacency_matrix_record))

def calc_population_C_rate(C_ratio_list):
    ratio_list = C_ratio_list[:, 0]
    total_number_list = C_ratio_list[:, 1]
    number_C = np.sum(np.multiply(ratio_list, total_number_list))
    total_number = np.sum(total_number_list)
    return round(number_C/total_number, 2)

# set parameters
num_agent = 20
steps = 20
rounds = 50
edge_prob = 0.2
run_spec = f'{num_agent}_{steps}_{rounds}_{edge_prob}'
intervention_types = ['RL', 'last_action', 'agent_ratio', 'network_ratio', 'randomized']
intervention_type = 'last_action'

# Load settings
prompts_file = json.load(open('settings/prompts.json'))
env_settings = json.load(open('settings/env_settings.json'))

a2c_manager = A2C_manager(state_dim=15, action_dim=3)
llm_agent_list = [LLMAgent(env_settings, 'llm_agent'+str(i)) for i in range(num_agent)]

a2c_manager.load_model(f'save_model/manager_networks_20nodes_40')

initial_time = time.time()
print(f'num_agent, steps, rounds, edge_prob: {run_spec}')

prompt_option_no_history_template = prompts_file['prompt_no_history']
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

whole_observation_list_record = []
whole_agent_ratio_list_record = []
whole_intervention_list_record = []
whole_adjacency_matrix_record = []

# outer loop for rounds
for episode in range(rounds):

    # creating the network
    g = connected_erdos_renyi_graph(n=num_agent, p=edge_prob)

    observation_list = -np.ones((num_agent, num_agent, 2))

    link_list = list(g.edges())
    C_ratio_list = np.zeros((num_agent, 2)) # the first value in second dim is the C_ratio, the second value is the number of actions.
    intervention_list = -np.ones((num_agent, num_agent, 2))

    round_observation_list_record = []
    round_agent_ratio_list_record = []
    round_intervention_list_record = []

    for step in range(steps):
        for (index1, index2) in link_list:
            # no RL intervention in first 2 steps
            # because we always get DD in the first step with "no history" prompt.
            if step < 2:
                prompt_1, prompt_2, intervention1, intervention2 = create_prompt(intervention_type, g, C_ratio_list, observation_list, index1, index2, step)
                
                action_1_str, action_2_str = get_LLM_response(llm_agent_list, prompt_1, prompt_2)

                observation_list[index1, index2] = [convert_str_to_int(action_1_str), convert_str_to_int(action_2_str)]
                observation_list[index2, index1] = [convert_str_to_int(action_2_str), convert_str_to_int(action_1_str)]

            else:
                prompt_1, prompt_2, intervention1, intervention2 = create_prompt(intervention_type, g, C_ratio_list, observation_list, index1, index2, step)
                
                action_1_str, action_2_str = get_LLM_response(llm_agent_list, prompt_1, prompt_2)

            C_ratio_list = update_C_ratio_list(C_ratio_list, index1, index2, action_1_str, action_2_str)

            observation_list[index1, index2] = [convert_str_to_int(action_1_str), convert_str_to_int(action_2_str)]
            observation_list[index2, index1] = [convert_str_to_int(action_2_str), convert_str_to_int(action_1_str)]

            intervention_list[index1, index2] = [intervention1, intervention2]
            intervention_list[index2, index1] = [intervention2, intervention1]

        # updating matrixes for each step
        round_agent_ratio_list_record.append(C_ratio_list)
        round_observation_list_record.append(observation_list)
        round_intervention_list_record.append(intervention_list)

        total_C_ratio = calc_population_C_rate(C_ratio_list)

        print(f'Round:{episode}, Step:{step}, C ratio: {total_C_ratio}')        

    # updating matrixes for each round
    whole_agent_ratio_list_record.append(round_agent_ratio_list_record)
    whole_observation_list_record.append(round_observation_list_record)
    whole_intervention_list_record.append(round_intervention_list_record)
    whole_adjacency_matrix_record.append(nx.adjacency_matrix(g).todense())

    print(f'Duration: {round(time.time() - initial_time, 2)}, Round:{episode}, '
          f'C ratio: {total_C_ratio}')

save_everything(whole_agent_ratio_list_record, whole_observation_list_record,
                whole_intervention_list_record, whole_adjacency_matrix_record, intervention_type, run_spec)