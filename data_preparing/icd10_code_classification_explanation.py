import pandas as pd
import json
import re
from tqdm import tqdm

def extract_descriptions(d, use_all_keys=False, parent_key='', results=None):
    """
    Extract descriptions from nested dictionaries.

    Parameters:
    - d: The dictionary to extract from.
    - use_all_keys: If True, extract all keys; otherwise, extract only 'description'.
    - parent_key: Used for recursive concatenation of keys.
    - results: Dictionary to store extracted results.

    Returns:
    - A dictionary of descriptions.
    """
    if results is None:
        results = {}
    
    for key, value in d.items():
        new_key = parent_key
        if isinstance(value, dict):
            extract_descriptions(value, use_all_keys, key, results)
        elif key == 'description' or (use_all_keys and key != 'description'):
            results[new_key.replace('.', '')] = value
    return results

icd_cm_file_path = 'MIMIC-IV-ICD-data-processing/crawl_ICD_final.json'
icd_pcs_file_path = 'MIMIC-IV-ICD-data-processing/crawl_ICD_PCS.json'
data_file_path = 'MIMIC-IV-ICD-data-processing/mimicdata/mimic4_icd10/disch_train_split.csv'

with open(icd_cm_file_path, 'r') as f:
    ICD_json_CM_list = extract_descriptions(json.load(f))

with open(icd_pcs_file_path, 'r') as f:
    ICD_json_PCS_list = extract_descriptions(json.load(f), use_all_keys=True)

df = pd.read_csv(data_file_path)

result_dict = {
    "Contributors": "mimic4_icd10 authors",
    "Source": "mimic4_icd10",
    "URL": "https://github.com/thomasnguyen92/MIMIC-IV-ICD-data-processing",
    "Categories": ["Text Classification"],
    "Definition": [
        "In medical practice, a practical task is to classify a discharge report into multiple ICD labels. "
        "Please recall the ICD-10 knowledge basis to understand all ICD-CM labels."
    ],
    "Reasoning": [],
    "Input_language": ["English"],
    "Output_language": ["English"],
    "Instruction_language": ["English"],
    "Domains": ["Medicine", "Clinical Reports", "Diagnosis"],
    "Positive Examples": [],
    "Negative Examples": [],
    "Instances": []
}

for index in tqdm(range(len(df)), desc="Processing classification data"):
    sample = df.iloc[index]
    input_text = sample["text"]
    output_labels = sample["labels"]
    ICD_no_list = output_labels.split(';')
    output_list = []

    for code in ICD_no_list:
        description = ICD_json_PCS_list.get(code) or ICD_json_CM_list.get(code)
        if description:
            output_list.append(f"{code}: {description}")
        else:
            print(f"Code not found: {code}")

    output_text = '; '.join(output_list)
    result_dict["Instances"].append({
        "input": input_text,
        "output": output_text
    })

with open("task107_mimic4_icd10_text_classification.json", 'w', encoding='utf-8') as f:
    json.dump(result_dict, f, ensure_ascii=False, indent=4)

result_dict = {
    "Contributors": "icd10-cm",
    "Source": "icd10-cm",
    "URL": "https://www.icd10data.com/ICD10PCS/Codes",
    "Categories": ["Explanation"],
    "Definition": [
        "In medical practice, ICD-10 is a classical coding basis. The task is to provide explanations for ICD-10 codes."
        "Each input code requires a detailed description of its medical meaning."
    ],
    "Reasoning": [],
    "Input_language": ["English"],
    "Output_language": ["English"],
    "Instruction_language": ["English"],
    "Domains": ["Medicine", "Clinical Knowledge"],
    "Positive Examples": [],
    "Negative Examples": [],
    "Instances": []
}

for code, description in {**ICD_json_CM_list, **ICD_json_PCS_list}.items():
    result_dict["Instances"].append({
        "input": code,
        "output": description
    })

with open("task108_icd10_code_explanation.json", 'w', encoding='utf-8') as f:
    json.dump(result_dict, f, ensure_ascii=False, indent=4)
