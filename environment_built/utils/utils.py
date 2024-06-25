import copy
import numpy as np

def get_variable_initial_value(variable_data):
    distribution = variable_data['distribution']

    if distribution == 'constant':
        initial_value = variable_data['args']['value']
    elif distribution == 'uniform':
        low = variable_data['args']['low']
        high = variable_data['args']['high']
        type = variable_data['args']['type']
        initial_value = np.random.uniform(low, high) if type == 'float' else int(np.random.uniform(low, high))
    elif distribution == 'normal':
        mean = variable_data['args']['mean']
        std = variable_data['args']['std']
        type = variable_data['args']['type']            
        initial_value = np.random.normal(mean, std) if type == 'float' else int(np.random.normal(mean, std))
    elif distribution == 'binomial':
        n = variable_data['args']['n']
        p = variable_data['args']['p']
        type = variable_data['args']['type']  
        initial_value = np.random.binomial(n, p) if type == 'float' else int(np.random.binomial(n, p))
    elif distribution == 'poisson':
        lam = variable_data['args']['lam']
        type = variable_data['args']['type']  
        initial_value = np.random.poisson(lam) if type == 'float' else int(np.random.poisson(lam))
    elif distribution == 'exponential':
        scale = variable_data['args']['scale']
        type = variable_data['args']['type']  
        initial_value = np.random.exponential(scale) if type == 'float' else int(np.random.exponential(scale))
    elif distribution == 'lognormal':
        mean = variable_data['args']['mean']
        std = variable_data['args']['std']
        type = variable_data['args']['type']  
        initial_value = np.random.lognormal(mean, std) if type == 'float' else int(np.random.lognormal(mean, std))
    elif distribution == 'gamma':
        shape = variable_data['args']['shape']
        scale = variable_data['args']['scale']
        type = variable_data['args']['type']  
        initial_value = np.random.gamma(shape, scale) if type == 'float' else int(np.random.gamma(shape, scale))
    elif distribution == 'categorical':
        options = list(variable_data['args'].keys())
        p = list(variable_data['args'].values())
        initial_value = np.random.choice(options, p=p)

    return initial_value

def get_action_initial_value(action_data):
    return

def merge_dicts(dict1, dict2):
    dict1 = copy.deepcopy(dict1)
    dict2 = copy.deepcopy(dict2)
    for key, value in dict2.items():
        if key in dict1:
            for sub_key, sub_value in value.items():
                if sub_key in dict1[key]:
                    dict1[key][sub_key].update(sub_value)
                else:
                    dict1[key][sub_key] = sub_value
        else:
            dict1[key] = value
    return dict1
    