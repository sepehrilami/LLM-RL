import ast

class FunctionHandlers:
    def __init__(self, logger=None):
        self.logger = logger

    def handle_function_call(self, family, agent, variable, function_call, data, global_data, is_memory=False):
        self.family = family
        self.agent = agent
        self.variable = variable
        self.data = data
        self.global_data = global_data
        self.is_memory = is_memory

        func_name, self.condition = self.parse_function_call(function_call)
        if hasattr(self, func_name):
            method = getattr(self, func_name)
            values, keys = self.filter_values()
            return method(values, keys)
        else:
            message = f"Function '{func_name}' is not supported"
            self.logger.error(message) if self.logger else print(message); exit()

    def parse_function_call(self, function_call):
        function_name = function_call.split('(')[0]
        condition = function_call[len(function_name) + 1:-1]  # Remove parentheses
        return function_name, condition

    def filter_values(self):
        if self.is_memory:
            if not self.condition:
                values = self.data[self.family][self.variable][self.agent]["memory"] 
                keys = list(range(1, len(values) + 1))
                return values, keys
            # values = [value for value in self.data[self.family][self.variable][self.agent]["memory"] if self.evaluate_condition()]
            # keys = [key for key in range(len(self.data[self.family][self.variable][self.agent]["memory"])) if self.evaluate_condition()]
            # return values, keys
            message = f"Conditions are not supported for memory functions"
            self.logger.error(message) if self.logger else print(message); exit()
        else:
            if not self.condition:
                values = [agent["value"] for agent in self.data[self.family][self.variable].values()]
                return values, None
            values = [agent_data["value"] for agent, agent_data in self.data[self.family][self.variable].items() if self.evaluate_condition()]
            return values, None

    def evaluate_condition(self):
        # Parse and evaluate the condition safely using ast and restricted globals
        variables = set()
        class VariableVisitor(ast.NodeVisitor):
            def visit_Name(self, node):
                variables.add(node.id)

        tree = ast.parse(self.condition, mode='eval')
        VariableVisitor().visit(tree)

        # Prepare local variables for evaluation based on the condition's variables
        local_vars = {}
        for var in variables:
            if var in self.data[self.family]:
                local_vars[var] = self.data[self.family][var][self.agent]["value"]
            else:
                message = f"Variable '{var}' not found in '{self.family}'"
                self.logger.error(message) if self.logger else print(message); exit()
        
        # Safe evaluation of the condition
        try:
            return eval(compile(tree, filename="<ast>", mode="eval"), {'__builtins__': {}}, local_vars)
        except Exception as e:
            message = f"Error evaluating condition '{self.condition}': {str(e)}"
            self.logger.error(message) if self.logger else print(message); exit()

    def mean(self, values, keys=None):
        try:
            return sum(values) / len(values)
        except Exception as e:
            message = f"Error calculating mean for '{self.family}.{self.variable}': {str(e)}"
            self.logger.error(message) if self.logger else print(message); exit()

    def sum(self, values, keys=None):
        if values:
            try:
                return sum(values)
            except Exception as e:
                message = f"Error calculating sum for '{self.family}.{self.variable}': {str(e)}"
                self.logger.error(message) if self.logger else print(message); exit()
        else:
            message = f"Error calculating sum for '{self.family}.{self.variable}': No values found"
            self.logger.error(message) if self.logger else print(message); exit()

    def min(self, values, keys=None):
        if values:
            try:
                return min(values)
            except Exception as e:
                message = f"Error calculating min for '{self.family}.{self.variable}': {str(e)}"
                self.logger.error(message) if self.logger else print(message); exit()
        else:
            message = f"Error calculating min for '{self.family}.{self.variable}': No values found"
            self.logger.error(message) if self.logger else print(message); exit()

    def max(self, values, keys=None):
        if values:
            try:
                return max(values)
            except Exception as e:
                message = f"Error calculating max for '{self.family}.{self.variable}': {str(e)}"
                self.logger.error(message) if self.logger else print(message); exit()
        else:
            message = f"Error calculating max for '{self.family}.{self.variable}': No values found"
            self.logger.error(message) if self.logger else print(message); exit()

    def count(self, values, keys=None):
        return len(values)
    
    def show(self, values, keys):
        step_unit = self.global_data["step_unit"]
        return ", ".join(f"{step_unit} {k}: {v}" for k, v in zip(keys, values))