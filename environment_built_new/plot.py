"""
This script is a utility to plot data from an output .csv file. 
The script can handle different scenarios based on the type of data.
"""


# Path to the .csv file
file_path = 'output/family_x.csv'

# A variable from the .csv file you want to plot
y_axis = "money_value"

# The name of the agent to plot the data for. If None, the mean of all agents is plotted or the categorical data is plotted
agent_name = None  



""" Plotting Utility """
import matplotlib.pyplot as plt
import pandas as pd

# Load the data
data = pd.read_csv(file_path)

x_axis = 'orbit_step'  # The default x-axis is 'orbit_step'

plt.figure(figsize=(12, 8))

if agent_name is None:
    # Get the type of value for the data in the y_axis
    y_type = data[y_axis].dtype

    # Scenario 1: numeric data
    if y_type == 'int64' or y_type == 'float64':
        data = data.groupby('orbit_step')[y_axis].mean().reset_index()
        plt.plot(data[x_axis], data[y_axis], marker='o')
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.title(f'{y_axis} vs {x_axis}')
        plt.grid(True)
        plt.show()

    # Scenario 2: categorical data
    elif y_type == 'object':
        data = data.groupby(['orbit_step', y_axis]).size().unstack().reset_index()
        ax = plt.gca()  # Get current axis to plot on
        data.plot(x='orbit_step', kind='line', ax=ax)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.title(f'Categorical {y_axis} vs {x_axis}')
        plt.grid(True)
        plt.show()
else:
    # Get the data for the specified agent
    data = data[data['agent_name'] == agent_name]

    x_data = data[x_axis]
    y_data = data[y_axis]

    # Get the type of value for the data in the y_axis
    y_type = data[y_axis].dtype

    # Scenario 1: numeric data
    if y_type == 'int64' or y_type == 'float64':
        plt.plot(data[x_axis], data[y_axis])
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.title(f'{y_axis} vs {x_axis}')
        plt.grid(True)
        plt.show()

    # Scenario 2: categorical data
    elif y_type == 'object':
        plt.scatter(data[x_axis], data[y_axis], c='r')  # Using red color for distinction
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.title(f'Categorical {y_axis} vs {x_axis}')
        plt.yticks(rotation=45)  # Rotate labels for better readability if needed
        plt.grid(True)
        plt.show()