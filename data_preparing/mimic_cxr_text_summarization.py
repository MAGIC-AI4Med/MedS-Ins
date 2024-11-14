"""
@Description :   Generate JSON file for SuperInstruction Text Summarization Task on MIMIC-CXR dataset.
@Author      :   Henrychur
"""

import os
import json
import csv
from tqdm import tqdm

def initialize_result_dict():
    """
    Initializes the result dictionary with metadata for the text summarization task.

    Returns:
    - A dictionary with metadata and structure for storing instances.
    """
    return {
        "Contributors": "MIMIC-CXR",
        "Source": "MIMIC-CXR",
        "URL": "https://physionet.org/content/mimic-cxr-jpg/2.0.0/",
        "Categories": ["Text Summarization"],
        "Definition": [
            "You will be given a description of the important aspects in the Chest X-ray image. Your task is to give a short summary of the most immediately relevant findings."
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

def extract_findings_and_impression(report_text):
    """
    Extracts findings and impression sections from the report text.

    Parameters:
    - report_text: The full text of the report.

    Returns:
    - A tuple containing findings content and impressions content.
    """
    findings_index = report_text.find("FINDINGS:")
    impressions_index = report_text.find("IMPRESSION:")

    findings_content = report_text[findings_index + len("FINDINGS: "):impressions_index].replace('\n', ' ').strip()
    impressions_content = report_text[impressions_index + len("IMPRESSION: "):].replace('\n', ' ').strip()

    return findings_content, impressions_content

def generate_text_summarization_data(input_csv, data_root, output_json):
    """
    Generates text summarization data for MIMIC-CXR and saves it to a JSON file.

    Parameters:
    - input_csv: Path to the CSV file containing report paths.
    - data_root: Root directory for report files.
    - output_json: Path to the output JSON file.
    """
    result_dict = initialize_result_dict()
    
    # Read data from CSV and extract instances
    with open(input_csv, 'r', encoding='utf-8') as fp:
        csv_reader = csv.DictReader(fp)
        
        for line in tqdm(csv_reader, desc="Processing reports"):
            report_path = os.path.join(data_root, line['report_path'])
            
            # Ensure the report file exists before proceeding
            if not os.path.exists(report_path):
                print(f"Warning: Report file {report_path} not found.")
                continue
            
            with open(report_path, 'r', encoding='utf-8') as report_file:
                report_text = report_file.read()
            
            # Check if both 'FINDINGS:' and 'IMPRESSION:' sections exist in the report
            if 'FINDINGS:' in report_text and 'IMPRESSION:' in report_text:
                findings_content, impressions_content = extract_findings_and_impression(report_text)
                
                # Add instance to result dictionary
                result_dict["Instances"].append({
                    "input": findings_content,
                    "output": impressions_content
                })
    
    # Save result dictionary to JSON file
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # Define input and output paths
    input_csv_path = 'MIMIC-CXR/cxr-study-list.csv'
    data_root = 'MIMIC-CXR'
    output_json_file = 'mimic_cxr_text_summarization.json'
    
    # Generate summarization data
    generate_text_summarization_data(input_csv_path, data_root, output_json_file)
