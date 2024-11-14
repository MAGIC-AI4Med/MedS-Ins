"""
@Description :   Generate JSON files for MedNLI textual entailment tasks.
@Author      :   Henrychur 
@Time        :   2024/03/22 15:46:55
"""

import json
import random
from tqdm import tqdm

# 定义输入和输出路径
ROOT_DIR = "mednli/1.0.0/mli_train_v1.jsonl"
SAVE_DIR = "./"

def initialize_result_dict(contributors, source, url, categories, definition, domains):
    """
    Initializes the result dictionary with metadata and basic structure.

    Parameters:
    - contributors: The contributors of the dataset.
    - source: The source of the dataset.
    - url: URL for more information.
    - categories: List of categories for the task.
    - definition: Instruction definition for the task.
    - domains: List of domains for the task.

    Returns:
    - A dictionary with basic metadata and structure for storing instances.
    """
    return {
        "Contributors": contributors,
        "Source": source,
        "URL": url,
        "Categories": categories,
        "Definition": definition,
        "Reasoning": [],
        "Input_language": ["English"],
        "Output_language": ["English"],
        "Instruction_language": ["English"],
        "Domains": domains,
        "Positive Examples": [],
        "Negative Examples": [],
        "Instances": []
    }

def load_mednli_data(filepath):
    """
    Loads MedNLI data from a JSONL file.

    Parameters:
    - filepath: Path to the JSONL file.

    Returns:
    - A list of dictionaries representing each line of data.
    """
    with open(filepath, 'r') as f:
        return [json.loads(line) for line in f]

def create_discriminative_task(data, output_file):
    """
    Creates a discriminative task for textual entailment.

    Parameters:
    - data: List of MedNLI data.
    - output_file: File path to save the output JSON.
    """
    result_dict = initialize_result_dict(
        contributors="mednli authors",
        source="mednli",
        url="https://jgc128.github.io/mednli/",
        categories=["Textual Entailment"],
        definition=["In medical, a practical task is to perform textual inference. In the following, you will be first "
                    "presented with a formal clinical premise statement. It may be some condition descriptions or "
                    "numerical results. Then I will give you a hypothesis statement, you have to determine whether the "
                    "hypothesis statement can be inferred from the clinical conditions. The two statements will be "
                    "formatted as 'sentence1: ...\\n\\nsentence2: ...'. Choose one: entailment, contradiction, or neutral."],
        domains=["Medicine", "Clinical Reports"]
    )
    
    for item in tqdm(data, desc="Creating Discriminative Task"):
        input_text = f"Sentence1: {item['sentence1']}\n\nSentence2: {item['sentence2']}"
        output_text = item["gold_label"].capitalize()
        result_dict["Instances"].append({"input": input_text, "output": output_text})
    
    with open(output_file, 'w') as f:
        json.dump(result_dict, f, indent=4)

def create_generative_task(data, output_file, label_filter, description):
    """
    Creates a generative task for textual entailment with specified label.

    Parameters:
    - data: List of MedNLI data.
    - output_file: File path to save the output JSON.
    - label_filter: Filter data based on this label (e.g., 'entailment' or 'contradiction').
    - description: Task description for the generative task.
    """
    result_dict = initialize_result_dict(
        contributors="mednli authors",
        source="mednli",
        url="https://jgc128.github.io/mednli/",
        categories=["Textual Entailment"] if label_filter == "entailment" else ["Wrong Candidate Generation"],
        definition=[description],
        domains=["Medicine", "Clinical Reports"]
    )
    
    for item in tqdm(data, desc=f"Creating Generative Task ({label_filter})"):
        if item["gold_label"] == label_filter:
            input_text = item["sentence1"].strip().capitalize()
            output_text = item["sentence2"].strip().capitalize() + "."
            result_dict["Instances"].append({"input": input_text, "output": output_text})
    
    with open(output_file, 'w') as f:
        json.dump(result_dict, f, indent=4)

def create_classification_task(data, output_file):
    """
    Creates a classification task with three sentences to classify.

    Parameters:
    - data: List of MedNLI data.
    - output_file: File path to save the output JSON.
    """
    result_dict = initialize_result_dict(
        contributors="mednli authors",
        source="mednli",
        url="https://jgc128.github.io/mednli/",
        categories=["Textual Entailment"],
        definition=["In following, you will first be given a medical premise statement. Then you will be provided with "
                    "three statements. You must determine which is entailment, which is contradiction, and which is neutral."],
        domains=["Medicine", "Clinical Reports"]
    )
    
    for i in tqdm(range(0, len(data) - 2, 3), desc="Creating Classification Task"):
        premise = data[i]["sentence1"].strip().capitalize()
        statements = random.sample([data[i], data[i + 1], data[i + 2]], 3)
        input_text = f"Statement: {premise}"
        output_text = ""
        
        for idx, statement in enumerate(statements, start=1):
            sentence = statement["sentence2"].strip().capitalize()
            label = statement["gold_label"]
            input_text += f"\nSentence{idx}: {sentence}"
            output_text += f"Sentence{idx} is {label}.\n"
        
        result_dict["Instances"].append({"input": input_text, "output": output_text})
    
    with open(output_file, 'w') as f:
        json.dump(result_dict, f, indent=4)

if __name__ == "__main__":
    # Load data
    data = load_mednli_data(ROOT_DIR)
    
    # Create different tasks
    create_discriminative_task(data, "task41_mednli_textual_entailment_discriminative.json")
    create_generative_task(data, "task42_mednli_textual_entailment_generative.json", "entailment",
                           "Your task is to perform textual entailment. You will be presented with a formal clinical premise statement and must write a hypothesis statement based on it.")
    create_generative_task(data, "task43_mednli_wrong_textual_entailment_generative.json", "contradiction",
                           "Your task is to generate a clearly incorrect hypothesis statement based on the provided premise.")
    create_classification_task(data, "task44_mednli_textual_entailment_classification.json")
