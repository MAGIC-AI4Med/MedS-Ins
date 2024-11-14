"""
@Description :   Generate JSON file for SuperInstruction Text Summarization Task on biomedical research articles.
@Author      :   Henrychur
"""

import json
from tqdm import tqdm

def initialize_result_dict():
    """
    Initializes the result dictionary with metadata for the text summarization task.

    Returns:
    - A dictionary with metadata and structure for storing instances.
    """
    return {
        "Contributors": "Readibility",
        "Source": "Readibility",
        "URL": "https://www.nactem.ac.uk/readability/",
        "Categories": ["Text Summarization"],
        "Definition": [
            "You will be given a biomedical research paper. Your task is to give a short, non-technical summary of the article."
        ],
        "Reasoning": [],
        "Input_language": ["English"],
        "Output_language": ["English"],
        "Instruction_language": ["English"],
        "Domains": ["Public Health", "Healthcare"],
        "Positive Examples": [],
        "Negative Examples": [],
        "Instances": []
    }

def parse_plos_corpus(filepath):
    """
    Parses the PLOS corpus JSONL file to create instances for text summarization.

    Parameters:
    - filepath: Path to the JSONL file containing the PLOS dataset.

    Returns:
    - A list of dictionaries where each dictionary contains 'input' and 'output' keys for summarization.
    """
    instances = []
    
    with open(filepath, 'r', encoding='utf-8') as fp:
        for line in tqdm(fp, desc="Processing PLOS Corpus"):
            json_data = json.loads(line.strip())
            title = json_data.get('title', '')
            article = json_data.get('article', '')
            pls = json_data.get('plain language summary', '')

            # Exclude PLS text from the article to avoid redundancy
            input_text = f"{title}\n{article.replace(pls, '')}"
            
            instances.append({
                "input": input_text,
                "output": pls
            })
    return instances

def save_to_json(data, output_file):
    """
    Saves the provided data to a JSON file.

    Parameters:
    - data: The data dictionary to save.
    - output_file: Path to the output JSON file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def generate_text_summarization_data(input_jsonl, output_json):
    """
    Generates text summarization data and saves it to a JSON file.

    Parameters:
    - input_jsonl: Path to the input JSONL file containing the dataset.
    - output_json: Path to the output JSON file.
    """
    result_dict = initialize_result_dict()
    result_dict["Instances"] = parse_plos_corpus(input_jsonl)
    save_to_json(result_dict, output_json)

if __name__ == "__main__":
    # Define input and output paths
    input_jsonl_file = 'plos_corpus/train_plos.jsonl'
    output_json_file = 'readibility_text_summarization.json'
    
    # Generate summarization data
    generate_text_summarization_data(input_jsonl_file, output_json_file)
