import time

from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import torch
import transformers
from torch import bfloat16


class Provider:
    def __init__(self, provider, model, temperature, api_key=None, max_query_attempts=20):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.client = self.get_provider()

        self.max_query_attempts = max_query_attempts

    def get_provider(self):
        if self.provider == 'groq':
            client = ChatGroq(
                model_name=self.model,
                temperature=self.temperature, 
                groq_api_key=self.api_key, 
                # model_kwargs={"response_format": {"type": "json_object"}}
                model_kwargs={}
            )
        elif self.provider == 'ollama':
            client = ChatOllama(
                model=self.model,
                temperature=self.temperature,
                format="json",
                num_gpu=1
            )
        else:
            print(f'Provider {self.provider} is not supported.')
            exit()
        return client
    
    def query_llm(self, prompt):
        format_instructions = ""
        attempt = 0
        flag = True
        messages = []
        messages.insert(0, SystemMessage(content=f"{prompt}\n\n{format_instructions}"))
        # print(messages)
        while flag:
            flag = False
            try:
                response = self.client.invoke(messages)
                if response.content != "C" and response.content != "D":
                    # print(response.content)
                    attempt += 1
                    flag = True
                    print(f"attempt: {attempt}, respond nonsense")
                    time.sleep(5)
            except:
                print(f"attempt: {attempt}, groq broke")
                attempt += 1
                flag = True
                time.sleep(5)

            if attempt >= 20:
                print("attempt more than tolerance, terminate the process")
                exit()
        # print(response)
        return response

