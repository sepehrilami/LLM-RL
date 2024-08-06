import numpy as np

def convert_int_to_str(action):

    if action == 0:
        return "D"
    elif action == 1:
        return "C"
    else:
        print(f'Invalid action int to str: {action}')
        return -1

def convert_str_to_int(action):
    if action == "D":
        return 0
    elif action == "C":
        return 1
    else:
        print(f'Invalid action str to int: {action}')
        return -1

def frequence_number_to_str(action):
    pass

def frequence_number_to_index(action):
    if action < 0.33:
        return [1, 0, 0]
    elif 0.66 > action and action >= 0.33:
        return [0, 1, 0]
    else:
        return [0, 0, 1]

def frequence_str_to_index(action):
    pass

def frequence_index_to_str(action):
    array = np.array(action)
    index = np.where(array == 1)[0][0]

    if index == 0:
        return "rarely"
    elif index == 1:
        return "sometime"
    else:
        return "often"


def get_reward(action1, action2):
    if action1 == "C":
        if action2 == "C":
            return ((3+3)-2)/(6-2)
        else:
            return ((0+5)-2)/(6-2)
    else:
        if action2 == "C":
            return ((5+0)-2)/(6-2)
        else:
            return ((1+1)-2)/(6-2)

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

def get_C_ratio(all_actions):
    np_actions = np.array(all_actions)
    C_ratio = round(np.sum(np_actions == "C") / len(all_actions), 2)

    return C_ratio