from datasets import load_dataset
import json

dataset = load_dataset("truehealth/liveqa")

data = dataset['train']


output_data = {
    "Contributors": "liveqa authors",
    "Source": "liveqa",
    "URL": "https://huggingface.co/datasets/truehealth/liveqa",
    "Categories": [
        "Question Answering",
        "Dialogue Generation"
    ],
    "Definition": [
        "Given your background as a doctor, please provide your insight in addressing the medical questions based on the patient's account."
    ],
    "Reasoning": [],
    "Input_language": [
        "English"
    ],
    "Output_language": [
        "English"
    ],
    "Instruction_language": [
        "English"
    ],
    "Domains": [
        "Medicine",
        "Daily Conversation"
    ],
    "Positive Examples": [],
    "Negative Examples": [],
    "Instances": []
}


for example in data:
    instance = {
        "input": example['message'], 
        "output": example['answer']   
    }
    output_data['Instances'].append(instance)


with open("task62_liveqa_medical_filtered_conversation.json", "w") as f:
    json.dump(output_data, f, indent=4)

