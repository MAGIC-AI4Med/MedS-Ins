import pandas as pd
import json
import re
from tqdm import tqdm

# 初始化结果字典
result_dict = {
    "Contributors": "Chaoyi Wu",
    "Source": "MIMIC-IV-Notes",
    "URL": "https://www.physionet.org/content/mimic-iv-note/2.2/",
    "Categories": ["Text Completion", "Data to Text"],
    "Definition": [
        "Discharge summaries are long-form narratives which describe the reason for a patient's admission to the hospital, "
        "their hospital course, and any relevant discharge instructions. They are professional documents, often hard to understand for the general public. "
        "In the following task, I will provide a discharge summary. Your goal is to complete the final 'Discharge Instructions' section, which is written for the patient."
    ],
    "Reasoning": [],
    "Input_language": ["English"],
    "Output_language": ["English"],
    "Instruction_language": ["English"],
    "Domains": ["Medicine", "Clinical Knowledge", "Disease"],
    "Positive Examples": [],
    "Negative Examples": [],
    "Instances": []
}

def remove_unnecessary_newlines(text):
    """
    Cleans up unnecessary newlines in the text while preserving structure.
    Specifically retains newlines following a colon, and removes others that are within sentences.
    """
    # Temporarily mark colons followed by newlines
    text = re.sub(r':\n', ':<COLON_NEWLINE>', text)
    
    # Remove newlines where they are followed by lowercase letters or specific punctuation
    text = re.sub(r'\n([a-z,_])', r'\1', text)
    text = re.sub(r'\n\s+([a-z])', r' \1', text)
    text = re.sub(r'\n([,;:\'\"().])', r'\1', text)
    
    # Restore newlines after colons
    text = text.replace(':<COLON_NEWLINE>', ':\n')
    return text

def process_discharge_data(file_path, output_path):
    """
    Processes discharge summaries from a CSV file and prepares them for text completion tasks.
    
    :param file_path: Path to the CSV file containing discharge summaries.
    :param output_path: Path where the resulting JSON file will be saved.
    """
    data = pd.read_csv(file_path)
    
    for i in tqdm(range(len(data)), desc="Processing summaries"):
        discharge_text = data.iloc[i]['text']
        discharge_summaries = remove_unnecessary_newlines(discharge_text)
        
        # Check if "Discharge Instructions:" section exists
        if "Discharge Instructions:\n" not in discharge_summaries:
            continue
        
        # Split text into input and output sections
        try:
            input_ss, output_ss = discharge_summaries.split("Discharge Instructions:\n", 1)
            output_ss = "Discharge Instructions:\n" + output_ss
            result_dict["Instances"].append({
                "input": input_ss,
                "output": output_ss
            })
        except Exception as e:
            print(f"Error processing summary at index {i}: {e}")
            continue
    
    # Save results to a JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    input_file_path = 'DATA/MIMIC-IV/mimic-iv-note/2.2/note/discharge.csv'
    output_file_path = "task76_mimic_iv_note_discharge_instruction_completion.json"
    process_discharge_data(input_file_path, output_file_path)
