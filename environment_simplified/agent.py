from utils.provider import Provider
from collections import deque

class LLMAgent():
    def __init__(self, setting_file, agent_name):

        self.setting = setting_file
        self.agent_name = agent_name

        self.provider_name = self.setting['provider']
        self.model = self.setting['model']
        self.temperature = self.setting['temperature']
        self.api_key = self.setting['api_key']

        self.provider = Provider(
            provider=self.provider_name,
            model=self.model,
            temperature=self.temperature,
            api_key=self.api_key,
        )

        self.memory_length = self.setting['memory_length']
        self.memory = {
            'prompt': deque(maxlen=self.memory_length),
            'llm_actions': deque(maxlen=self.memory_length)
        }
            
    def answer(self, prompt):
        # Update memory
        self.memory['prompt'].append(prompt)

        # Query llm
        llm_actions = self.provider.query_llm(prompt)

        self.memory['llm_actions'].append(llm_actions.content)

        return llm_actions.content
