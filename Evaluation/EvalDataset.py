from typing import (
    AbstractSet,
    cast,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Sequence,
    TypedDict,
    Union,
)
from tiktoken.load import load_tiktoken_bpe
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import json
import PIL
import numpy as np
import torch.nn.functional as F
import tqdm
import transformers
import os
import copy
import random
import jsonlines
import pandas as pd
import csv
import random
from typing import Dict, Optional, Sequence

IGNORE_INDEX = -100

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


Dialog = Sequence[Message]

WITH_RATIONAL = False

DEFAULT_PAD_TOKEN = "</s>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


class ChatFormat:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode_header(self, message: Message) -> List[int]:
        strr = "<|start_header_id|>"
        strr = strr + message["role"] + "<|end_header_id|>" + "\n\n"
        return strr

    def encode_message(self, message: Message) -> List[int]:
        strr = self.encode_header(message)
        strr = strr + message["content"].strip()+"<|eot_id|>"
        return strr

    def encode_dialog_prompt(self, dialog: Dialog) -> List[int]:
        strr = "<|begin_of_text|>"
        for message in dialog:
            strr = strr + self.encode_message(message)
        strr = strr + self.encode_header({"role": "assistant", "content": ""})
        return strr

class InsDataset(Dataset):
    def __init__(self, tokenizer, root_path, definition_prompt='./RewriteDefinitions.json', original_ins_possibility=1, no_context_possibility=1, model_type='Llama 3'):
        task_list = os.listdir(root_path)
        print('The Trained Task list is: \n', task_list, '\n\n')
        self.Pos_Ins_task_dic = {}
        with open(definition_prompt, 'r') as f:
            self.Task_definition = json.load(f)
        self.Instances = []
        self.original_ins_possibility = original_ins_possibility
        self.no_context_possibility = no_context_possibility
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.ntasks = 1
        self.reflect = {}
        for task_json in task_list:
            Ins_tt_id = task_json.split('_')[0]
            self.reflect[Ins_tt_id] = self.ntasks
            self.ntasks = self.ntasks + 1
            with open(root_path + '/' + task_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
                Instances = data["Instances"]
                for index in range(0, len(Instances)):
                    self.Instances.append(
                        {
                            'task_id': Ins_tt_id,
                            "input": Instances[index]["input"],
                            "output": Instances[index]["output"]
                        }
                    )
            try:
                self.Task_definition[Ins_tt_id]['original_def'] = data["Definition"]
            except:
                self.Task_definition[Ins_tt_id] = {
                    'original_def': data["Definition"],
                    "auto_rewrite_def": data["Definition"]
                }
            self.Pos_Ins_task_dic[Ins_tt_id] = Instances

    def __len__(self):
        return len(self.Instances)

    def __getitem__(self, idx):
        case = self.Instances[idx]
        task_id = case['task_id']
        Pos_instance = self.Pos_Ins_task_dic[task_id]

        q = random.random()
        if q <= self.original_ins_possibility:
            Instruction = self.Task_definition[task_id]['original_def'][0]
        else:
            try:
                Instruction = random.sample(
                    self.Task_definition[task_id]["auto_rewrite_def"], 1)[0]
            except:
                Instruction = self.Task_definition[task_id]['original_def'][0]

        p = random.random()
        if p <= self.no_context_possibility:
            query = case["input"]
        else:
            if True:
                sample_number = 3
                Pos_instance = random.sample(
                    Pos_instance, min(len(Pos_instance), sample_number))
                try:
                    few_shot_cases = ["Input:\n" + _['input'] + "\n" + '\n' +
                                      "Output:\n" + _['output'] + "\n" for _ in Pos_instance]
                except:
                    few_shot_cases = ["Input:\n" + _['input'] + "\n" + '\n' +
                                      "Output:\n" + _['output'][0] + "\n" for _ in Pos_instance]
                few_shot_string = '\n\n'.join(few_shot_cases) + '\n\n'
                Instruction = few_shot_string + Instruction + \
                    'Please learn from the few-shot cases to see what content you have to output.'
                query = "Input:\n" + case["input"] + "\n" + "Output:\n"

        if self.model_type == 'Llama 3':
            Input_sentence = self.build_inputs_for_llama3(query, Instruction)
        elif self.model_type == 'MMedLlama 3':
            Input_sentence = self.build_inputs_for_llama3(query, Instruction)
        elif self.model_type == 'InternLM 2':
            Input_sentence = self.build_inputs_for_internlm(query, Instruction)
        elif self.model_type == 'Mistral':
            Input_sentence = self.build_inputs_for_mistral(query, Instruction)
        elif self.model_type == 'GPT-4':
            Input_sentence = self.build_inputs_for_GPT(query, Instruction)
        elif self.model_type == 'Claude':
            Input_sentence = self.build_inputs_for_GPT(query, Instruction)
        else:
            Input_sentence = self.default_build_inputs(query, Instruction)

        if isinstance(case["output"], list):
            output_sentence = case["output"][0]
        else:
            output_sentence = case["output"]
        output_sentence = output_sentence
        return {
            'task_id': task_id,
            'input': Input_sentence,
            'output': output_sentence
        }

    def build_inputs_for_internlm(self, query: str, instruction=""):
        if self.tokenizer.add_bos_token:
            prompt = ""
        else:
            prompt = self.tokenizer.bos_token
        if instruction:
            prompt += f"""<|im_start|>system\n{instruction}<|im_end|>\n"""
        prompt += f"""<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"""
        return prompt

    def build_inputs_for_llama3(self, query: str, instruction=""):

        input_ss = [
            Message(role='system', content=instruction),
            Message(role='user', content=query)
        ]
        Prompt_engine = ChatFormat(tokenizer=self.tokenizer)
        return Prompt_engine.encode_dialog_prompt(input_ss)

    def build_inputs_for_mistral(self, query: str, instruction=""):

        input_ss = [
            {"role": "user", "content": instruction + '\n' + query},
        ]
        return input_ss

    def build_inputs_for_GPT(self, query: str, instruction=""):
        return {
            'system': instruction,
            'user': query
        }

    def default_build_inputs(self, query: str, instruction=""):
        if self.tokenizer.add_bos_token:
            prompt = ""
        else:
            prompt = self.tokenizer.bos_token
        if instruction:
            prompt += instruction+'\n'
        prompt += query
        return prompt
