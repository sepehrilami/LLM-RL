# Holodeck
Holodeck is an Agent-Based Modeling (ABM) framework and tool that integrates Large Language Models (LLMs) into agents’ decision-making processes. Designed to facilitate the creation of diverse and intricate ABM scenarios, Holodeck enables users to study and explore complex systems. Distinct from conventional LLM frameworks that prioritize task-solving, Holodeck offers a fresh approach, enhancing the capabilities of standard ABM tools and broadening the scope of research and exploration in complex systems.

It is strongly reccomended to read the following **[overview](https://gabrigoo.notion.site/Holodeck-ffca2f4f6f7e42dba6ef92575f783cf5?pvs=4)** before utilizing Holodeck.


## Table of Contents
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Output](#output)
- [Notation Guide](#notation-guide)
- [Variable Initialization Distributions](#variable-initialization-distributions)
- [Providers](#providers)


## Installation
**Note**: It is recommended to install and run holodeck in a virtual environment
```
git clone https://github.com/gabriansa/holodeck.git
cd holodeck
pip install virtualenv
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


## Usage Guide
To successfully run a simulation with Holodeck, you need to set up two critical JSON configuration files in the ```settings``` directory: `orbit_settings.json` and `family_settings.json`. These files define the environment and behaviors of the agents within your simulation. Once defined, the simulation can be ran with the following command:
```
python main.py
```

[Holodeck Guide](holodeck_guide.md) is a simple step-by-step guide on how to design a simulation.

### Creating `orbit_settings.json`
This file outlines the global settings for the simulation's environment. Here is a detailed breakdown of what each field represents:

- **`orbit_name`**: A descriptive name for the simulation
- **`orbit_steps`**: The number of steps the simulation will run for
- **`update_type`**: Specifies how updates to the model are handled (`synchronous` or `asynchronous`)
- **`seed`**: An integer used to initialize the random number generator
- **`provider`**: The provider of the LLM service (see [Providers](#providers))
- **`model`**: Specific model identifier used for processing
- **`temperature`**: Controls randomness in the output generation (lower values produce more deterministic outputs)
- **`api_key`**: Authentication key required to access the LLM service (if applicable, else ```null```)

#### Example `orbit_settings.json`
```json
{
    "orbit_name": "Orbit 1",
    "orbit_steps": 403,
    "update_type": "synchronous",
    "seed": 0,
    "provider": "groq",
    "model": "llama3-70b-8192",
    "temperature": 0.5,
    "api_key": "your_api_key_here"
}
```

### Creating `family_settings.json`
This file configures the families of agents, including their behaviors, variables, and actions. Below is a guide on each segment within a family's configuration:

- **`family_name`**: The name identifying a group of agents
- **`number_of_agents`**: Specifies how many agents belong to this family
- **`memory_length`**: This specifies the number of past messages an agent can remeber (see Memory to explore more about how memory can be accessed)
- **`system_prompt_template`**: Template for llm system prompt, where variables can be inserted dynamically (see [Notation Guide](#notation-guide) on how to use variables)
- **`prompt_template`**: Template for interactions involving this family, where variables can be inserted dynamically (see [Notation Guide](#notation-guide) on how to use variables)

#### Variables Section
Defines the variables for each agent:
- **`variable_name`**: Name of the variable (no spaces allowed)
- **`distribution`**: How the variable's initial value is determined (see [Variable Initialization Distributions](#variable-initialization-distributions))
- **`args`**: Arguments relevant to the chosen distribution (e.g., fixed values or probabilities for categories, see [Variable Initialization Distributions](#variable-initialization-distributions))
- **`update_rule`**: This field specifies how the variable is updated each simulation step. If no update is required, you can set it to `null`. For simple updates, you can directly apply basic arithmetic operations. For example:
   ```json
   "update_rule": "self.variable_x + 1"
   ```
   This straightforward rule will increment the value of `variable_x` by 1 at each step of the simulation. For more complex conditions and dynamic behaviors, you can incorporate inline Python-like conditional statements:
   ```json
   "update_rule": "self.variable_x + 1 if self.variable_x < 100 else self.variable_x"
   ```
   Here, `variable_x` is incremented by 1 only if its current value is less than 100; otherwise, it remains the same.

#### Actions Section
Specifies possible actions agents can take:
- **`action_name`**: A unique identifier for the action
- **`action_description`**: A brief description of what the action entails
- **`action_type`**: The type of the action (e.g., "option", "number", "string")
- **`action_options`**: Relevant for "option" type actions, listing the possible choices 

#### Example `family_settings.json`
```json
[
    {
        "family_name": "family_x",
        "number_of_agents": 10,
        "memory_length": 1,
        "system_prompt_template": "This is a system prompt",
        "prompt_template": "Here is my variable x {self.variable_x}. Here is the mean of variable x for all agents in my family {family_x.variable_x.mean()}",
        "variables": [
            {
                "variable_name": "variable_x",
                "distribution": "constant",
                "args": {"value": 10.15},
                "update_rule": "self.variable_x + 1"
            },
            {
                "variable_name": "variable_y",
                "distribution": "uniform",
                "args": {"low": 1, "high": 10, "type": "int"},
                "update_rule": null
            },
            {
                "variable_name": "variable_z",
                "distribution": "categorical",
                "args": {"A": 0.2, "B": 0.8},
                "update_rule": null
            }
        ],
        "actions": [
            {
                "action_name": "option_action",
                "action_description": "this is a test action",
                "action_type": "option",
                "action_options": ["option_1", "option_2", "option_3"]
            },
            {
                "action_name": "number_action",
                "action_description": "this is a test action",
                "action_type": "number"
            },
            {
                "action_name": "text_action",
                "action_description": "this is a test action",
                "action_type": "string"
            }
        ]
    }
]
```

## Output
By default, the simulation output is stored in the `data` directory, unless specified otherwise. The contents of this directory include several important files, each serving a specific purpose:

- **`info.html`**: This file documents all interactions within the simulation, including the prompts issued and the responses received by each agent from each family. It is useful for a detailed review of agent behaviors and interactions
  
- **`info.log`**: This log file captures system operations and errors. If the simulation encounters any issues, details can be found here
  
- **`<FAMILY_NAME>.csv`**: For each family in the simulation, there is a corresponding `.csv` file named after the family. These files record the variables and actions of each agent for every step of the simulation. They are particularly valuable for data analysis and tracking the evolution of agent states over time


## Notation Guide
In Holodeck, calculations can happen at the prompt level or in the update rules. To include calculations and variables in the promppt use curly brackets `{}`. This section outlines the correct notation for referencing variables, families, and functions within the framework:

### Accessing Agent's Variables
- **`self.<VARIABLE_NAME>`** → Used to refer to the value of `<VARIABLE_NAME>` for a specific agent within a prompt or update rule

- **`<FAMILY_NAME>.<VARIABLE_NAME>.<AGENT_ID>`** → Utilized to obtain the value of `<VARIABLE_NAME>` for a particular `<AGENT_ID>` within `<FAMILY_NAME>` (Agent IDs are standard and are as follows `agent_0`, `agent_1`, etc.)

### Function Calling
- **`<FAMILY_NAME>.<VARIABLE_NAME>.<FUNCTION>(<CONDITION>)`** → Applied to execute a `<FUNCTION>` across all variables of agents in that family that meet a `<CONDITION>`. Functions available include:
  - `mean()`
  - `max()`
  - `min()`

- **`<FAMILY_NAME>.<FUNCTION>(<CONDITION>)`** → Applied to execute a `<FUNCTION>` across a family that meet a `<CONDITION>`. Functions available include:
  - `count()`

- **`<CONDITION>`** → To apply conditions, use relational operators (`<`, `>`, `==`) along with logical operators (`and`, `or`). For example, to calculate the mean income of agents over 50 years old with a disability in the "citizens" family, you would use: `citizens.income.mean(age > 50 and disability == 'yes')`
- **Handling `None` Values:** If a variable is `None`, any line in the prompt containing that variable within curly brackets `{}` will be omitted to prevent referencing data without value


### Memory
There are two ways to acess an agent's memory. 
1. The first way is to set the `memory_length` parameter on the `family_settings.json` file to specify the number of past messages an agent can remeber. This method can incur into slow simulation runs since at each step an agent will feed the past conversation in the LLM model.

2. **`self.<VARIABLE_NAME>.memory.<FUNCTION>()`** or **`<FAMILY_NAME>.<VARIABLE_NAME>.<AGENT_ID>.memory.<FUNCTION>()`** → Retrieves the memory for `<VARIABLE_NAME>` in a string-readible way depending on the `<FUNCTION>` used. This can be used in the prompt template. Available functions are:
    - `show()`: shows the memory in the following format `Step 1: <VALUE>, Step 2: <VALUE>, etc.`

## Variable Initialization Distributions
For initializing variables in the `family_settings.json` file, various distribution options are available, each requiring specific arguments. Here’s a guide to setting up these distributions correctly:

1. **Constant**:
   - **value**: The constant value assigned to the variable
   ```json
   "distribution": "constant",
   "args": {"value": 10}
   ```

2. **Uniform**:
   - **low**: The lower bound of the distribution
   - **high**: The upper bound of the distribution
   - **type**: Specifies the data type, either `'float'` or `'int'`
   ```json
   "distribution": "uniform",
   "args": {"low": 1, "high": 10, "type": "int"}
   ```

3. **Normal**:
   - **mean**: The mean of the distribution
   - **std**: The standard deviation of the distribution
   - **type**: `'float'` or `'int'`
   ```json
   "distribution": "normal",
   "args": {"mean": 0, "std": 1, "type": "float"}
   ```

4. **Binomial**:
   - **n**: Number of trials
   - **p**: Probability of success in each trial
   - **type**: `'float'` or `'int'`
   ```json
   "distribution": "binomial",
   "args": {"n": 10, "p": 0.5, "type": "int"}
   ```

5. **Poisson**:
   - **lam**: The expectation of interval (known as lambda)
   - **type**: `'float'` or `'int'`
   ```json
   "distribution": "poisson",
   "args": {"lam": 3, "type": "float"}
   ```

6. **Exponential**:
   - **scale**: The scale parameter of the distribution, inverse of the rate
   - **type**: `'float'` or `'int'`
   ```json
   "distribution": "exponential",
   "args": {"scale": 2, "type": "float"}
   ```

7. **Lognormal**:
   - **mean**: The mean of the underlying normal distribution
   - **std**: The standard deviation of the underlying normal distribution
   - **type**: `'float'` or `'int'`
   ```json
   "distribution": "lognormal",
   "args": {"mean": 0, "std": 1, "type": "float"}
   ```

8. **Gamma**:
   - **shape**: The shape parameter of the distribution
   - **scale**: The scale parameter of the distribution
   - **type**: `'float'` or `'int'`
   ```json
   "distribution": "gamma",
   "args": {"shape": 2, "scale": 2, "type": "float"}
   ```

9. **Categorical**:
   - This distribution requires an object with category names as keys and their respective probabilities as values
   ```json
   "distribution": "categorical",
   "args": {"A": 0.2, "B": 0.8}
   ```


## Providers
### Groq (Recommended)
1. **Create an Account**: Sign up at Groq by following this [link](https://console.groq.com/login)
2. **API Key**: After creating your account, generate an API key from the Groq console
3. **Model ID**: Use the model ID provided by Groq for the model name in the `orbit_settings.json`

### Ollama
1. **Installation**: Download Ollama software from the [official website](https://ollama.com/)
2. **Download Models**: Choose and download a model from the avialable models (see [Ollama Library](https://ollama.com/library))
3. **Model Name**: Once downloaded, you can retrieve the model name using the `ollama list` command in the terminal
4. **API Key**: Since Ollama models run locally, set the API key to `null` in the `orbit_settings.json`