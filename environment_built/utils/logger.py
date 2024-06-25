import os
import logging
import textwrap

class Logger(logging.Logger):
    def __init__(self, path='output', name='Logger', filename='info'):
        super().__init__(name, level=logging.INFO)
        self.path = path

        self.html_file_path = os.path.join(self.path, filename + '.html')
        self.log_file_path = os.path.join(self.path, filename + '.log')

        # Set up file handler
        file_handler = logging.FileHandler(self.log_file_path)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.addHandler(file_handler)
        self.propagate = False  # Prevents log messages from being propagated to the root logger

        # Set up empty html file
        html_header = '<html><head><style>pre {font-family: monospace;}</style></head><body><pre>\n'
        html_footer = '</pre></body></html>'
        with open(self.html_file_path, 'w') as f:
            f.write(html_header + html_footer)

    def agent_step(self, orbit_step, family_name, agent_name, system_prompt, prompt, llm_actions, width=120):
        border_color = "color:gray;"

        system_prompt_style = "<span style='color:green; font-weight:bold;'>"
        prompt_style = "<span style='color:magenta; font-weight:bold;'>"
        action_style = "<span style='color:red; font-weight:bold;'>"
        action_name_style = "<span style='color:red; font-weight:normal;'>"
        
        # Title, system prompt, prompt, and llm actions
        title = f"Orbit Step: {orbit_step}, Family: {family_name}, Agent: {agent_name}"
        system_prompt_text = self.wrap_text(system_prompt, width=width)
        prompt_text = self.wrap_text(prompt, width=width)

        action_text = {}
        for action_name, action_data in llm_actions.items():
            action_text[action_name] = {}
            action_text[action_name]['reasoning'] = self.wrap_text(action_data['reasoning'], width=width-2)
            action_text[action_name]['action'] = self.wrap_text(str(action_data['action']), width=width-2)

        # Create the HTML output
        html_output = []

        # Title section
        html_output.append(f"<span style='{border_color}'>╭─<span style='font-weight:bold;'>  {title}  </span>─{'─' * (width - len(title) - 4)}╮</span>")

        # System Prompt section
        html_output.append(f"<span style='{border_color}'>│{system_prompt_style}System Prompt</span> {' ' * (width - len('System Prompt') + 1)}│</span>")
        for line in system_prompt_text.split('\n'):
            line_formatted = " " + line + " " + " " * (width - len(line))
            html_output.append(f"<span style='{border_color}'>│</span>" + line_formatted + f"<span style='{border_color}'>│</span>")

        # Prompt section
        html_output.append(f"<span style='{border_color}'>│{prompt_style}Prompt</span> {' ' * (width - len('Prompt') + 1)}│</span>")
        for line in prompt_text.split('\n'):
            line_formatted = " " + line + " " + " " * (width - len(line))
            html_output.append(f"<span style='{border_color}'>│</span>" + line_formatted + f"<span style='{border_color}'>│</span>")

        # Actions section
        html_output.append(f"<span style='{border_color}'>│{action_style}Actions</span> {' ' * (width - len('Actions') + 1)}│</span>")
        for action_name, action_data in action_text.items():
            html_output.append(f"<span style='{border_color}'>│{action_name_style}{' ' +action_name}</span> {' ' * (width - len(action_name))}│</span>")
            for line in action_data['action'].split('\n'):
                line_formatted = "  " + line + " " + " " * (width - len(line)-1)
                html_output.append(f"<span style='{border_color}'>│</span>" + line_formatted + f"<span style='{border_color}'>│</span>")
            for line in action_data['reasoning'].split('\n'):
                line_formatted = "  " + line + " " + " " * (width - len(line)-1)
                html_output.append(f"<span style='{border_color}'>│</span>" + line_formatted + f"<span style='{border_color}'>│</span>")

        # Closing border
        html_output.append(f"<span style='{border_color}'>╰{'─' * (width+2)}╯</span>")

        # Spacer
        html_output.append("\n")

        # Write to HTML file, ensuring to maintain existing content
        with open(self.html_file_path, 'r') as f:
            lines = f.readlines()

        lines.insert(-1, "\n".join(html_output) + "\n")

        with open(self.html_file_path, 'w') as f:
            f.writelines(lines)
    
    def wrap_text(self, text, width):
        lines = text.split('\n')  # Split the text on each newline to maintain existing line breaks
        wrapped_lines = [textwrap.fill(line, width=width) for line in lines]  # Wrap each line separately
        wrapped_text = '\n'.join(wrapped_lines)  # Rejoin all wrapped lines into a single string with preserved newlines
        return wrapped_text

    def error(self, msg, *args, **kwargs):
        super().error(msg, *args, **kwargs)
        exit()    
