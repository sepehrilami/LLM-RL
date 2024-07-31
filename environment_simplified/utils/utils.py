import numpy as np

def convert_int_to_str(action):

    if action == 0:
        return "D"
    elif action == 1:
        return "C"
    else:
        print(f'Invalid action: {action}')
        return -1

def convert_str_to_int(action):
    if action == "D":
        return 0
    elif action == "C":
        return 1
    else:
        print(f'Invalid action: {action}')
        return -1

def get_reward(action):
    if action == "C":
        return 1
    elif action == "D":
        return 0
    else:
        print(f'Invalid action: {action}')
        return None

def stat_analysis(all_actions):
    # this list is a list of C and D actions.
    # count them and print the ration of C and D.
    np_actions = np.array(all_actions)
    # print(f'Total actions: {len(all_actions)}')
    print(f'Ratio of Y: %{round(np.sum(np_actions == "Y") / len(all_actions), 2) * 100}, P: %{round(np.sum(np_actions == "P") / len(all_actions), 2) * 100}')
    print("--------------------")

def get_C_ratio(all_actions):
    np_actions = np.array(all_actions)
    C_ratio = round(np.sum(np_actions == "C") / len(all_actions), 2)

    return C_ratio