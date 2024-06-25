## Step-by-Step Guide to Defining a Simulation in Holodeck

Creating an effective simulation in Holodeck involves a thoughtful and meticulous setup process. Below is a detailed, step-by-step guide to help you define a robust simulation environment tailored to your specific research or study needs.

### Step 0: Define Simulation Unit
- **Step Unit**: Identify what each step of the simulation means (i.e. a year, an hour, a decade, etc.). This is crucial to the properly design update rules for each variable.

### Step 1: Define Families
- **Families**: Determine how many families (or groups of agents) you need for your simulation. Families can represent different groups, entities, or categories depending on the scenario.
- **Agents**: Decide on the number of agents in each family. This could range from a single agent to hundreds or thousands, based on the complexity and scale of your simulation.

### Step 2: Define Variables for Each Family
- **Variables**: Identify which variables are crucial for each family. Variables could represent attributes like health, income,personality, etc.
- **Initial Values and Distributions**: Set initial values using appropriate distributions (e.g., constant, uniform, normal). Ensure these values reflect realistic scenarios for your simulation’s context. If using a distribution determine the correct distirbution arguments (i.e. mean, std, etc.).
- **Update Rules**: This field specifies how the variable is updated each simulation step. If no update is required, you can set it to `null` Update rules could be as simple as for instance `self.<VARIABLE_NAME> + 1` or more complex in terms of equations or logic such as `self.<VARIABLE_NAME> + 1 if self.<VARIABLE_NAME> < 100 else self.<VARIABLE_NAME>`. You can also consider updates that might involve operations between different variables, possibly across families.

### Step 3: Configure Actions for Each Family
- Determine what actions agents can take during the simulation. These might include movements, buying/selling, or interactions with other agents.
- **Action Types**: Define whether these are options (choices), numeric, or text-based.
- For each action attach a short description related to it.

### Step 4: Design Prompt Templates
- Develop prompt templates for each family that incorporate variables. These templates guide the interactions and decisions of agents.
- **Include Functions and Operations**: Integrate functions like `mean()` or `max()` to use aggregate information (e.g., average income of a family) in decision-making processes.
- **Embed Variables and Logic**: Utilize embedded variables and conditional logic within templates to reflect dynamic changes and decision criteria.

### Step 5: Put Everything Together
For each family come up with a document of a similar structure:
- **Family Name**: `<FAMILY_NAME>`
- **Number of Agents**: number of agents for the specified family
- **Variables**: for each variable define:
   - **Variable Name**: `<VARIABLE_NAME>`
   - **Initialization Type**: uniform, constant, categorical, binomial, etc.
   - **Arguments**: if it's a distribution mention the arguments of that distribution and their values. If it is a constant/text mention that value. If it is categorical mention the category names and their respective probabilities.
   - **Update Rule**: define how the variable updates for each step of the simulation. This could include simple and complex equations as well as logic operators and can make use of other families as well as actions.
- **Actions**: for each action define:
   - **Action Name**: `<ACTION_NAME>`
   - **Action Type**: this can be an option, a number, or some text. If it's an option please mention the possible options available (i.e. "buy", "sell", "hold").
   - **Action Description**: this is a concise description of the action.