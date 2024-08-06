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
        elif self.provider == 'hf':
            bnb_config = transformers.BitsAndBytesConfig(load_in_4bit=True,
                                                         bnb_4bit_quant_type='nf4',
                                                         bnb_4bit_use_double_quant=True, 
                                                         bnb_4bit_compute_dtype=bfloat16)
            model_config = transformers.AutoConfig.from_pretrained(self.model,token=self.api_key)
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.model, token=self.api_key)
            torch.cuda.empty_cache()
            alg = transformers.AutoModelForCausalLM.from_pretrained(self.model,
                                                                    trust_remote_code=True,
                                                                    config=model_config,
                                                                    quantization_config=bnb_config,
                                                                    device_map='cuda:0', token=self.api_key)
            self.pipe = transformers.pipeline(model=alg, tokenizer=tokenizer, torch_dtype=bfloat16, return_full_text=True,
                                              task='text-generation', temperature=self.temperature, repetition_penalty=1.1,
                                              pad_token_id=tokenizer.eos_token_id, batch_size=8)
            self.pipe.call_count = 0
            client = HuggingFacePipeline(pipeline=self.pipe,
                                         model_kwargs={"response_format": {"type": "json_object"}})
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

