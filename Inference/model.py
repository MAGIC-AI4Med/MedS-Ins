"""
@Description: LLM for inference
@Author: Henrychur 
@Time: 2024/08/26 15:59:20
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Literal, Sequence, TypedDict

Role = Literal["system", "user", "assistant"]

class Message(TypedDict):
    role: Role
    content: str

Dialog = Sequence[Message]

class ChatFormat:
    def encode_header(self, message: Message) -> str:
        return f"{message['role']}\n\n"

    def encode_message(self, message: Message) -> str:
        header = self.encode_header(message)
        return f"{header}{message['content'].strip()}"

    def encode_dialog_prompt(self, dialog: Dialog) -> str:
        dialog_str = ""
        for message in dialog:
            dialog_str += self.encode_message(message)
        dialog_str += self.encode_header({"role": "assistant", "content": ""})
        return dialog_str

class MedS_Llama3:
    def __init__(self, model_path: str, gpu_id: int = 0):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=f"cuda:{gpu_id}",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        self.model.config.pad_token_id = self.model.config.eos_token_id = 128009
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=2048,
            padding_side="right",
            trust_remote_code=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.prompt_engine = ChatFormat()
        self.model.eval()
        print('Model and tokenizer loaded!')

    def __build_inputs_for_llama3(self, query: str, instruction: str) -> str:
        input_ss = [
            {"role": 'system', "content": instruction},
            {"role": 'user', "content": query}
        ]
        return self.prompt_engine.encode_dialog_prompt(input_ss)

    def chat(self, history: List[tuple], query: str, instruction: str = "If you are a doctor, please perform clinical consulting with the patient.") -> str:
        if len(history) > 0:
            raise NotImplementedError("The model does not support multi-turn conversation.")
        
        formatted_query = f"Input:\n{query}\nOutput:\n"
        input_sentence = self.__build_inputs_for_llama3(formatted_query, instruction)
        
        input_tokens = self.tokenizer(
            input_sentence,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)

        output = self.model.generate(
            **input_tokens,
            max_new_tokens=2048,
            eos_token_id=128009
        )

        generated_text = self.tokenizer.decode(
            output[0][input_tokens['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )

        return generated_text.strip()

