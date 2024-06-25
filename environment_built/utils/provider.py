import json
from jsonschema import validate

from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

class Provider:
    def __init__(self, provider, model, temperature, api_key=None, max_query_attempts=20, logger=None):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.logger = logger

        self.client = self.get_provider()

        self.max_query_attempts = max_query_attempts

    def get_provider(self):
        if self.provider == 'groq':
            client = ChatGroq(
                model_name=self.model,
                temperature=self.temperature, 
                groq_api_key=self.api_key, 
                model_kwargs={"response_format": {"type": "json_object"}}
            )
        elif self.provider == 'ollama':
            client = ChatOllama(
                model=self.model,
                temperature=self.temperature,
                format="json"
            )
        else:
            message = f"Provider '{self.provider}' is not supported."
            self.logger.error(message)
        
        return client
    
    def query_llm(self, memory, memory_length, actions, family_name, agent_name):
        self.memory = memory
        self.memory_length = memory_length
        self.actions = actions
        self.family_name = family_name
        self.agent_name = agent_name

        messages = self.get_messages()
        # print(messages)
        response = self.query_logic(messages)
        # print(response)
        return response
    
    def query_logic(self, messages):
        query_attempts = 0
        while query_attempts <= self.max_query_attempts:
            query_attempts += 1
            try:
                response = self.client.invoke(messages)
                response = json.loads(response.content)
            except Exception as e:
                message = f"[{self.family_name}, {self.agent_name}] Attempt #{query_attempts} failed - Error querying the model '{self.model}' from provider '{self.provider}': {e}"
                self.logger.warning(message) if self.logger else print(message)
                continue

            for action, agents in self.actions[self.family_name].items():
                # print(agents.get(self.agent_name))
                allowed_keys = {
                    action: agents.get(self.agent_name)['options'] if agents.get(self.agent_name)['type'] == 'option' else agents.get(self.agent_name)['type']
                }
                # print(agents.get(self.agent_name)['type'])

            schema = self.generate_schema(allowed_keys)
            # print(response)
            # print(schema)
            if not self.validate_dict(response, schema):
                message = f"[{self.family_name}, {self.agent_name}] Attempt #{query_attempts} failed - The response is not in the correct format: {response}"
                self.logger.warning(message) if self.logger else print(message)
                continue
            else:
                return response

        message = f"[{self.family_name}, {self.agent_name}] Error querying the model '{self.model}' from provider '{self.provider}'. Max attempts exceeded."
        self.logger.error(message) if self.logger else print(message); exit()

        return response

    def validate_dict(self, data, schema):
        try:
            validate(instance=data, schema=schema)
            return True
        except:
            return False
    
    def generate_schema(self, allowed_keys):
        properties = {}
        for key, action_type in allowed_keys.items():
            # print(action_type)
            if isinstance(action_type, list):
                type_def = {"type": "string", "enum": action_type}
            elif action_type == "integer":
                type_def = {"type": "integer"}
            elif action_type == "list_action":
                type_def = {"type": "list"}
            elif action_type == "string":
                type_def = {"type": "string"}
            elif action_type == "float":
                type_def = {"type": "number"}
            elif action_type == "number":
                type_def = {"type": "number"}
            else:
                type_def = {"type": "null"}  # Default case if an unrecognized type is provided

            properties[key] = {
                "type": "object",
                "properties": {
                    "action": type_def,
                    "reasoning": {"type": "string"}
                },
                "required": ["action", "reasoning"],
                "additionalProperties": False
            }
        
        schema = {
            "type": "object",
            "properties": properties,
            "additionalProperties": False
        }
        return schema

    def get_messages(self):
        format_instructions = self.get_format_instructions()

        messages = []
        for i in range(len(self.memory['prompt'])):
            messages.append(HumanMessage(content=self.memory['prompt'][i]))
            if i < len(self.memory['llm_actions']):
                messages.append(AIMessage(content=self.memory['llm_actions'][i]))

        if len(messages)/2 > self.memory_length:
            messages = messages[-2*self.memory_length:]

        messages.insert(0, SystemMessage(content=f"{self.memory['system_prompt']}\n\n{format_instructions}"))
        messages.append(HumanMessage(content=self.memory['prompt'][-1]))
        # print(messages)
        return messages
        
    def get_format_instructions(self):
        format_instructions = "The response should have the following JSON format:\n{\n"

        for action, agents in self.actions[self.family_name].items():
            agent_data = agents.get(self.agent_name)
            if agent_data['type'] == 'text':
                format_instructions += f'\t\"{action}\": {{\n'
                format_instructions += "\t\t\"reasoning\": string // the reasoning behind the action\n"
                format_instructions += f"\t\t\"action\": string // {agent_data['description']}\n"
            elif agent_data['type'] == 'option':
                options = ', '.join(agent_data['options'])
                format_instructions += f"\t\"{action}\": {{\n"
                format_instructions += "\t\t\"reasoning\": string // the reasoning behind the chosen option\n"
                format_instructions += f"\t\t\"action\": option // select one option from the following options [{options}]\n"
            elif agent_data['type'] == 'float':
                format_instructions += f"\t\"{action}\": {{\n"
                format_instructions += "\t\t\"reasoning\": string // the reasoning behind the action\n"
                format_instructions += f"\t\t\"action\": float // {agent_data['description']}\n"
            elif agent_data['type'] == 'integer':
                format_instructions += f"\t\"{action}\": {{\n"
                format_instructions += "\t\t\"reasoning\": string // the reasoning behind the action\n"
                format_instructions += f"\t\t\"action\": integer // {agent_data['description']}\n"
            elif agent_data['type'] == 'number':
                format_instructions += f"\t\"{action}\": {{\n"
                format_instructions += "\t\t\"reasoning\": string // the reasoning behind the action\n"
                format_instructions += f"\t\t\"action\": number // {agent_data['description']}\n"
            elif agent_data['type'] == 'list':
                format_instructions += f"\t\"{action}\": {{\n"
                format_instructions += "\t\t\"reasoning\": string // the reasoning behind the action\n"
                format_instructions += f"\t\t\"action\": list // {agent_data['description']}\n"
        format_instructions += "}\n"

        return format_instructions