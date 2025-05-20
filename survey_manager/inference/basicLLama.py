import transformers
import torch

from vllm import LLM, SamplingParams

import gc

from vllm.sampling_params import GuidedDecodingParams

import random

from typing import List, Dict, Any

from pydantic import BaseModel

class BasicLLama:
    
    def __init__(self, model_id :str ="meta-llama/Meta-Llama-3.1-70B-Instruct"):
        random.seed(42)
        torch.manual_seed(42)
        self._model_id = model_id

        print("Device_count: " + str(torch.cuda.device_count()))

        self._generator = LLM(model=self._model_id, tensor_parallel_size=torch.cuda.device_count(), seed=42, max_model_len=7500, enable_prefix_caching=True)
    
    def batch_generation(self, system_messages:list[str]=["You are a helpful assistant."], prompts:list[str]=["Hi there! What is your name?"], max_new_tokens:int = 256, temperature = 0.6, top_p:float=0.9, seed: int = 42):
        # Prepare batch of messages
        batch_messages = [
        [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}]
        for system_message, prompt in zip(system_messages, prompts)
        ]

        batch_size = len(system_messages)

        seeds = [random.randint(0, 2**32 - 1) for _ in range(batch_size)]

        sampling_params_list = [
            SamplingParams(temperature=temperature, top_p=top_p, top_k=50, max_tokens=max_new_tokens, seed=seeds[i])
            for i in range(batch_size)
        ]

        outputs = self._generator.chat(batch_messages, sampling_params=sampling_params_list, use_tqdm=False,)
        result = [output.outputs[0].text for output in outputs]
        #print(result, flush=True)
        return result

        
    def batch_turn_by_turn_generation(self, system_messages:List[str]=["You are a helpful assistant."], prompts:List[List[str]]=[["Hi there! What is your name?", "Interesting"]], assistant_messages:List[List[str]]=None, json_structured_output:bool = False, json_schemas:List[Dict[str, Any]] = None, max_new_tokens:int = 256, temperature:float = 0.6, top_p:float=0.9) -> List[str]:
        batch_messages = []
        batch_size = len(system_messages)
        for i in range(batch_size):
            messages = []

            # Add system message
            if system_messages[i]:
                messages.append({"role": "system", "content": system_messages[i]})
                
            num_user_msgs = len(prompts[i])
            num_assistant_msgs = len(assistant_messages[i])

            for j in range(num_user_msgs):
                messages.append({"role": "user", "content": prompts[i][j]})
                if j < num_assistant_msgs:
                    messages.append({"role": "assistant", "content": assistant_messages[i][j]})

            batch_messages.append(messages)

        seeds = [random.randint(0, 2**32 - 1) for _ in range(batch_size)]
        guided_decoding_params_list = [ GuidedDecodingParams(json=json_schemas[i])
            for i in range(batch_size)  
        ]
        
        if json_structured_output:
            sampling_params_list = [
            SamplingParams(temperature=temperature, top_p=top_p, top_k=50, max_tokens=max_new_tokens, guided_decoding=guided_decoding_params_list[i], seed=seeds[i])
            for i in range(batch_size)
            ]
        else:
            sampling_params_list = [
                SamplingParams(temperature=temperature, top_p=top_p, top_k=50, max_tokens=max_new_tokens, seed=seeds[i])
                for i in range(batch_size)
            ]

        print(batch_messages, flush=True)

        outputs = self._generator.chat(batch_messages, sampling_params=sampling_params_list, use_tqdm=False,)
        result = [output.outputs[0].text for output in outputs]
        
        return result
    
    def batch_choice_structured_ouput(self, system_messages:list[str]=["You are a helpful assistant."], prompts:list[str]=["Hi there! What is your name?"], structured_ouput_options:List[List[str]] = [["I AM AN AI OVERLORD!", "I am just a nice lil' person."]], max_new_tokens:int = 256, temperature = 0.6, top_p:float=0.9, seed: int = 42) -> List[str]:

        # Prepare batch of messages
        batch_messages = [
        [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}]
        for system_message, prompt in zip(system_messages, prompts)
        ]

        batch_size = len(system_messages)

        seeds = [random.randint(0, 2**32 - 1) for _ in range(batch_size)]

        guided_decoding_params_list = [ GuidedDecodingParams(choice=structured_ouput_options[i])
            for i in range(batch_size)  
        ]

        print(guided_decoding_params_list)

        sampling_params_list = [
            SamplingParams(temperature=temperature, top_p=top_p, top_k=50, max_tokens=max_new_tokens, guided_decoding=guided_decoding_params_list[i], seed=seeds[i])
            for i in range(batch_size)
        ]

        outputs = self._generator.chat(batch_messages, sampling_params=sampling_params_list, use_tqdm=False,)
        result = [output.outputs[0].text for output in outputs]
        #print(result, flush=True)
        return result
    

    def batch_json_structured_ouput(self, system_messages:list[str]=["You are a helpful assistant."], prompts:list[str]=["Hi there! What is your name?"], json_schemas:List[Dict[str, Any]] = None, max_new_tokens:int = 256, temperature = 0.6, top_p:float=0.9, seed: int = 42) -> List[str]:

        # Prepare batch of messages
        batch_messages = [
        [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}]
        for system_message, prompt in zip(system_messages, prompts)
        ]

        batch_size = len(system_messages)

        seeds = [random.randint(0, 2**32 - 1) for _ in range(batch_size)]

        guided_decoding_params_list = [ GuidedDecodingParams(json=json_schemas[i])
            for i in range(batch_size)  
        ]

        print(guided_decoding_params_list)

        sampling_params_list = [
            SamplingParams(temperature=temperature, top_p=top_p, top_k=50, max_tokens=max_new_tokens, guided_decoding=guided_decoding_params_list[i], seed=seeds[i])
            for i in range(batch_size)
        ]

        outputs = self._generator.chat(batch_messages, sampling_params=sampling_params_list, use_tqdm=False,)
        result = [output.outputs[0].text for output in outputs]
        #print(result, flush=True)
        return result
    
    def shutdown(self):
        # Explicitly delete LLM and call GC
        del self._generator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()