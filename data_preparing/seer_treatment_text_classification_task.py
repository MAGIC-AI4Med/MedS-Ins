"""
@Description :   Generate JSON file for SEER dataset text classification task on treatment planning.
@Author      :   Henrychur 
@Time        :   2024/03/22 15:46:55
"""

import pandas as pd
import json
from tqdm import tqdm

def initialize_result_dict():
    """
    Initializes the result dictionary with metadata for the SEER text classification dataset.

    Returns:
    - A dictionary with metadata and structure for storing instances.
    """
    return {
        "Contributors": "DDXPlus authors",
        "Source": "DDXPlus",
        "URL": "https://figshare.com/articles/dataset/DDXPlus_Dataset/20043374",
        "Categories": ["Text Classification"],
        "Definition": [
            "Imagine you are a doctor. Next, I will provide a summary of a patient, and you need to suggest the next treatment. "
            "Please choose from: ['Intraoperative rad with other rad before/after surgery', 'Intraoperative radiation', "
            "'No radiation and/or cancer-directed surgery', 'Radiation after surgery', 'Radiation before and after surgery', "
            "'Radiation prior to surgery', 'Surgery both before and after radiation']"
        ],
        "Reasoning": [],
        "Input_language": ["English"],
        "Output_language": ["English"],
        "Instruction_language": ["English"],
        "Domains": ["Medicine", "Text Classification", "Treatment Planning"],
        "Positive Examples": [],
        "Negative Examples": [],
        "Instances": []
    }

def extract_instances(data, feature_columns, output_column):
    """
    Processes each row in the dataset to create instances for the classification task.

    Parameters:
    - data: DataFrame containing the dataset.
    - feature_columns: List of column names to include in the input.
    - output_column: Name of the column containing the output label.

    Returns:
    - A list of instances where each instance contains formatted input and output data.
    """
    instances = []
    for index in tqdm(range(len(data)), desc="Processing instances"):
        sample = data.iloc[index]
        # Generate input text with key-value pairs from specified columns
        input_text = "\n\n".join([f"{key}: {sample[key]}" for key in feature_columns])
        output_text = f"The treatment planning is: {sample[output_column]}"
        
        instances.append({
            "input": input_text,
            "output": output_text
        })
    return instances

def save_to_json(data, output_file, train_ratio=0.9):
    """
    Saves a portion of the data to a JSON file.

    Parameters:
    - data: Dictionary containing all instances.
    - output_file: Path to the output JSON file.
    - train_ratio: Proportion of the data to save for training.
    """
    num_train = int(train_ratio * len(data["Instances"]))
    data["Instances"] = data["Instances"][:num_train]

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # Define input and output paths
    root_path = 'SEER_treatment_and_alive_prediction/Query_5_years.csv'
    output_file = "task131_SEER_text_classification_train.json"
    
    # Load data
    df = pd.read_csv(root_path)
    
    # Define feature columns and output column
    feature_columns = [col for col in df.columns if col not in ['Radiation sequence with surgery', 'stutus_5_years']]
    output_column = 'Radiation sequence with surgery'
    
    # Initialize result dictionary and populate instances
    result_dict = initialize_result_dict()
    result_dict["Instances"] = extract_instances(df, feature_columns, output_column)

    # Save to JSON file
    save_to_json(result_dict, output_file)
