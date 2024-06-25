import re

from utils.function_handlers import FunctionHandlers

class ExpressionEvaluator:
    def __init__(self, logger=None):
        self.logger = logger
        self.function_handlers = FunctionHandlers(logger)

    def evaluate(self, expression, data, global_data, family_name, agent_name, literal):
        self.data = data
        self.global_data = global_data
        self.family_name = family_name
        self.agent_name = agent_name

        if literal:
            output = []
            lines = expression.split('\n')
            for line in lines:
                expression_ = re.compile(r'{(.*?)}')
                matches = expression_.finditer(line)
                self.none_detected = False
                for match in matches:
                    result = self.evaluate_expression(match.group(1))
                    # line = line.replace(match.group(0), str(result)) if not self.none_detected else ""
                    line = line.replace(match.group(0), str(result))

                output.append(line)
            return '\n'.join(output)
        else:
            self.none_detected = False
            expression = self.evaluate_expression(expression)
            if self.none_detected:
                return ""
            return expression

    def evaluate_expression(self, expression):
        pattern = self.create_pattern()
        equation = re.sub(pattern, self.parser, expression)

        try:
            result = eval(compile(equation, filename="<ast>", mode="eval"), {'__builtins__': {}})
            return int(result) if isinstance(result, int) else int(result) if isinstance(result, float) and result.is_integer() else round(result, 2) if isinstance(result, float) else str(result)
        except Exception as e:
            message = f"Error evaluating equation '{expression}' --> '{equation}': {str(e)}"
            self.logger.error(message) if self.logger else print(message); exit()

    def create_pattern(self):
        base_patterns = [
            r'\bself\.\w+\.\w+\.\w+\((?:[^()]*|\([^)]*\))*\)',       # Matches self.word.word.word() for self.<VARIABLE_NAME>.<MEMORY>.<FUNCTION_NAME>(<CONDITION>)
            r'\bself\.\w+',                                          # Matches self.word for self.<VARIABLE_NAME>
            r'\b\w+\.\w+\.\w+\.\w+\.\w+\((?:[^()]*|\([^)]*\))*\)',   # Matches word.word.word.word.word() for <FAMILY_NAME>.<VARIABLE_NAME>.<AGENT_NAME>.<MEMORY>.<FUNCTION_NAME>(<CONDITION>)
            r'\b\w+\.\w+\.\w+\((?:[^()]*|\([^)]*\))*\)',             # Matches word.word.word() with nested optional for <FAMILY_NAME>.<VARIABLE_NAME>.<FUNCTION_NAME>(<CONDITION>)
            r'\b\w+\.\w+\.\w+',                                      # Matches word.word.word for <FAMILY_NAME>.<VARIABLE_NAME>.<AGENT_NAME>
        ]

        extra_items = list(self.global_data.keys())

        patterns = [r'\b' + re.escape(item) for item in extra_items] + base_patterns
        final_pattern = r'|'.join(patterns)

        return re.compile(final_pattern)
              
    def parser(self, match):
        parts = match.group(0).split(".")
        value = self.handle_expression(parts)
        return str(f'"{value}"') if isinstance(value, str) else str(value)

    def handle_expression(self, parts):
        if parts[0] in list(self.global_data.keys()):
            return self.global_data[parts[0]]
        elif parts[0] == "self":
            family_name = self.family_name
            agent_name = self.agent_name
            idx = 2
        elif parts[0] in self.data:
            family_name = parts[0]
            agent_name = parts[2]
            idx = 3
        else:
            message = f"Syntax error: '{'.'.join(parts)}' is not a valid expression"
            self.logger.error(message) if self.logger else print(message); exit()

        variable_name = parts[1]

        if variable_name in self.data[family_name]:
            if agent_name in self.data[family_name][variable_name]:
                value = self.data[family_name][variable_name][agent_name]["value"]
                if len(parts) > idx:
                    if "memory" == parts[idx] and parts[idx+1].endswith(")"):
                        # Edge case with action being None at the beginning
                        values = self.data[family_name][variable_name][agent_name]["memory"]
                        condition = parts[idx+1]
                        if len(values) == 0:
                            self.none_detected = True
                            return 0
                        else:
                            return self.function_handlers.handle_function_call(family_name, agent_name, variable_name, condition, self.data, self.global_data, is_memory=True)
                    else:
                        message = f"Syntax error: '{'.'.join(parts)}' is not a valid expression"
                        self.logger.error(message) if self.logger else print(message); exit()
                elif value is None:
                    self.none_detected = True
                    return 0
                else:
                    return value
            elif parts[idx-1].endswith(")"):
                # Edge case with action being None at the beginning
                values = [agent["value"] for agent in self.data[family_name][variable_name].values()]
                if None in values:
                    self.none_detected = True
                    return 0
                condition = parts[idx-1]
                return self.function_handlers.handle_function_call(family_name, "", variable_name, condition, self.data, self.global_data)
            
        message = f"Syntax error: '{'.'.join(parts)}' is not a valid expression"
        self.logger.error(message) if self.logger else print(message); exit()