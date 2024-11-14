"""
@Description :   Generate JSON file for SuperInstruction Text Summarization Task.
@Author      :   Henrychur
"""

import json
import pandas as pd
from tqdm import tqdm

def initialize_result_dict():
    """
    Initializes the result dictionary with metadata for the text summarization task.

    Returns:
    - A dictionary with metadata and structure for storing instances.
    """
    return {
        "Contributors": "MeQSum",
        "Source": "MeQSum",
        "URL": "https://github.com/abachaa/MeQSum",
        "Categories": ["Text Summarization"],
        "Definition": [
            "You will be given a long medical question. Your task is to summarize the consumer health question."
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

def generate_text_summarization_data(input_file, output_file):
    """
    Generates text summarization data from an Excel file and saves it to a JSON file.

    Parameters:
    - input_file: Path to the input Excel file containing the dataset.
    - output_file: Path to the output JSON file.
    """
    # Load dataset from Excel
    try:
        dataset = pd.read_excel(input_file)
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return
    
    # Initialize result dictionary
    result_dict = initialize_result_dict()
    
    # Populate instances
    for _, row in tqdm(dataset.iterrows(), desc="Processing rows", total=dataset.shape[0]):
        input_text = row['CHQ']
        summary = row['Summary']
        
        # Add instance to the result dictionary
        result_dict["Instances"].append({
            "input": input_text,
            "output": summary
        })
    
    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # Define input and output paths
    input_xlsx_file = 'MeQSum_ACL2019_BenAbacha_Demner-Fushman.xlsx'
    output_json_file = 'medqsum_text_summarization.json'
    
    # Generate summarization data
    generate_text_summarization_data(input_xlsx_file, output_json_file)
