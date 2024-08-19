# Digital Twins of Group Dynamics: Enhancing Pro-Social Behavior with LLM Agents and Adaptive RL Governance

## Installing the requirements
You can install the requirements using
```pip install -r requirements.txt```

## Code
All codes needed for reproducing the results are available in the main_code directory. All codes needed for viualizing the results and creating figures for the paper are provided in the create_figures directory.

## Set hyper-parameters
If you wish to change the hyper-parameters of the experiment, you can modify them in this file.
The default hyper-parameters for the environment are as below:
num_agent = 20, steps = 20, rounds = 10, edge_prob = 0.25

### Rerun the experiments
You can run the rl_test_network.py file to create the network of agents and let them play and interact with each other. The framework will save the results for every round. NOTE: You need to use your own Groq API in the /settings.env_setting.json file to be able to use Groq platform for loading the LlaMa3 model.

### Description of each files
rl_train_network.py: The training phase of the Reinforcement Learning model.

rl_test_network.py: The main file for the experiment. You can evaluate the perfomance of different methods (LA, LA+NR, LA+AR, RL) which are explained detailed in the main paper.

micro_validation.py: This file validates the behavior of an agent at the micro-level based on the information given to them in the prompt.

agent.py: There are two types of agents in this framework: LLM Agents and RL Agent. The Strategic LLM Agents (SLAs) are created by LLM Agents which can make decisions based on given information in the prompts. The RL agent, called Prompting Prosocial Agent (PPA), is the moderator of the information and decides how to choose the information properly to increase the social welfare of the whole system. 

Prompts are avalable in the settings/prompts.json file. 

## Figures

The visualizer.ipynb and plot_evolution_pairactions.py files create the figures for the main paper and for the SI paper.

All of the figures are available in the /figs directory.