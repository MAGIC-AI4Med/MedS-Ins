"""
@Description :   Generate JSON data for critical triage prediction from MIMIC-IV EHR dataset.
@Author      :   Henrychur 
@Time        :   2024/03/22 15:46:55
"""

import pandas as pd
import json
from tqdm import tqdm

def initialize_result_dict():
    """
    Initializes the result dictionary with metadata for the critical triage dataset.

    Returns:
    - A dictionary with metadata structure and an empty list for storing instances.
    """
    return {
        "Contributors": "mimic4ed-benchmark authors",
        "Source": "mimic4ed-benchmark",
        "URL": "https://github.com/nliulab/mimic4ed-benchmark",
        "Categories": ["Data to Text"],
        "Definition": [
            "Next I will provide EHR data of a patient. Your goal is to predict whether the patient should be classified "
            "into the critical triage queue. This helps clinicians prioritize high-risk cases."
        ],
        "Reasoning": [],
        "Input_language": ["English"],
        "Output_language": ["English"],
        "Instruction_language": ["English"],
        "Domains": ["Medicine", "EHR", "Triage", "Emergency Department"],
        "Positive Examples": [],
        "Negative Examples": [],
        "Instances": []
    }

def extract_instances(data, variables, outcome_col):
    """
    Processes each row in the dataset to create instances for prediction tasks.

    Parameters:
    - data: DataFrame containing the dataset.
    - variables: List of variable names to include in each instance's input.
    - outcome_col: Name of the column containing the outcome label.

    Returns:
    - A list of instances where each instance contains formatted input and output data.
    """
    instances = []
    for i in tqdm(range(len(data)), desc="Processing instances"):
        sample = data.iloc[i]
        # Generate input string with key-value pairs from the specified variables
        input_text = '\n'.join([f"{key}: {sample[key]}" for key in variables])
        output_text = str(sample[outcome_col])

        instances.append({
            "input": input_text,
            "output": output_text
        })
    return instances

def save_to_json(result_dict, output_file):
    """
    Saves the result dictionary to a JSON file.

    Parameters:
    - result_dict: The dictionary containing the data to save.
    - output_file: Path to the output JSON file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # Define input file path and output file name
    input_file_path = 'mimic4ed-benchmark-main/master_dataset.csv'
    output_file_path = "task105_mimic4ed_benchmark_critical_triage.json"
    
    # Define variable columns to be included in the input data
    variables = [
        "age", "gender",
        "n_ed_30d", "n_ed_90d", "n_ed_365d", "n_hosp_30d", "n_hosp_90d", "n_hosp_365d",
        "n_icu_30d", "n_icu_90d", "n_icu_365d", "triage_temperature", "triage_heartrate", 
        "triage_resprate", "triage_o2sat", "triage_sbp", "triage_dbp", "triage_pain", 
        "triage_acuity", "chiefcom_chest_pain", "chiefcom_abdominal_pain", 
        "chiefcom_headache", "chiefcom_shortness_of_breath", "chiefcom_back_pain", 
        "chiefcom_cough", "chiefcom_nausea_vomiting", "chiefcom_fever_chills", 
        "chiefcom_syncope", "chiefcom_dizziness", "cci_MI", "cci_CHF", "cci_PVD", 
        "cci_Stroke", "cci_Dementia", "cci_Pulmonary", "cci_Rheumatic", "cci_PUD", 
        "cci_Liver1", "cci_DM1", "cci_DM2", "cci_Paralysis", "cci_Renal", "cci_Cancer1", 
        "cci_Liver2", "cci_Cancer2", "cci_HIV", "eci_Arrhythmia", "eci_Valvular", 
        "eci_PHTN", "eci_HTN1", "eci_HTN2", "eci_NeuroOther", "eci_Hypothyroid", 
        "eci_Lymphoma", "eci_Coagulopathy", "eci_Obesity", "eci_WeightLoss", 
        "eci_FluidsLytes", "eci_BloodLoss", "eci_Anemia", "eci_Alcohol", "eci_Drugs",
        "eci_Psychoses", "eci_Depression"
    ]
    outcome_column = "outcome_critical"

    # Load data
    data = pd.read_csv(input_file_path)
    
    # Initialize result dictionary and populate instances
    result_dict = initialize_result_dict()
    result_dict["Instances"] = extract_instances(data, variables, outcome_column)

    # Save to JSON file
    save_to_json(result_dict, output_file_path)
