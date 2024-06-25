class DataHandler:
    def __init__(self, path='output'):
        self.path = path
 
    def collect(self, orbit_step, variables, actions, family_name, agent_name, llm_actions):
        # Define the header
        header = ['orbit_step', 'agent_name']
        for variable_name, variable_data in variables[family_name].items():
            header.append(f"{variable_name}_value")
        for action_name, action_data in actions[family_name].items():
            header.append(f"{action_name}_value")
        
        # Define the data
        data = [orbit_step, agent_name]
        for variable_name, variable_data in variables[family_name].items():
            data.append(variable_data[agent_name]["value"])

        for action_name, action_data in llm_actions.items():
            # print(action_data)
            data.append(action_data["actions"])
            # data.append(action_data)

        # Append the data to the csv
        with open(f'{self.path}/{family_name}.csv', 'a') as f:
            # if the file is empty, write the header
            if f.tell() == 0:
                f.write(','.join(header) + '\n')
            f.write(','.join(map(str, data)) + '\n')