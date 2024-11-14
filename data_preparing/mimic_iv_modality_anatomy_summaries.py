"""
@Description :   Extract data for SFT from MIMIC-IV
@Author      :   Henrychur
@Time        :   2024/03/27 16:45:38
"""
import os
import csv
import json
from tqdm import tqdm

# 定义模式和解剖部位分类
MODALITY = ['X-ray', 'CT', 'MRI', 'Ultrasound', 'Angiography', 'Mammogram', 'Pathology', 'Fluoroscopy']
ANATOMY = ['Chest', 'Head_and_neck', 'Brain', 'Abdomen', 'Pelvis', 'Spine']

def get_modality(extracted_text):
    """
    Extract modality indices based on keywords in the text.
    """
    res_list = []
    lower_text = extracted_text.lower()
    for i, modality in enumerate(MODALITY):
        if modality.lower() in lower_text:
            res_list.append(i)
    return res_list

def get_anatomy(extracted_text):
    """
    Extract anatomy index based on keywords in the text.
    """
    lower_text = extracted_text.lower()
    for i, anatomy in enumerate(ANATOMY):
        if anatomy.lower() in lower_text:
            return i
    return -1

def extract_findings_and_impressions(text):
    """
    Extract findings and impressions sections from the text.
    """
    findings_index = text.find("FINDINGS:")
    impressions_index = text.find("IMPRESSION:")
    findings_content = text[findings_index + len("FINDINGS: "):impressions_index].strip()
    impressions_content = text[impressions_index + len("IMPRESSION: "):].strip()
    return findings_content, impressions_content

def process_modality_summaries(folder_path, filename):
    """
    Process data to generate summaries based on modalities and save as JSON files.
    """
    res_data = [[] for _ in range(len(MODALITY))]
    with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as fp:
        csv_dict_reader = csv.DictReader(fp)
        for row in tqdm(csv_dict_reader, desc="Processing modalities"):
            text = row['text']
            if all(keyword in text for keyword in ['FINDINGS:', 'IMPRESSION:', 'EXAMINATION:', 'INDICATION:']):
                findings_content, impressions_content = extract_findings_and_impressions(text)
                start_index = text.find("EXAMINATION:") + len("EXAMINATION:")
                end_index = text.find("INDICATION:")
                extracted_text = text[start_index:end_index].strip()
                modalities = get_modality(extracted_text)
                
                for modality_id in modalities:
                    res_data[modality_id].append({
                        'input': findings_content,
                        'output': impressions_content
                    })

    save_summaries(res_data)

def save_summaries(res_data):
    """
    Save summaries as JSON files based on modality.
    """
    for idx, data in enumerate(res_data):
        if idx >= 3:  # Skip if modality is X-ray, CT, or MRI
            definition = f"Given the detailed finding of {MODALITY[idx]} imaging diagnostics, summarize the note's conclusion in a few words."
            res_json_dict = {
                'Contributors': 'MIMIC',
                'Source': 'MIMIC-IV',
                'URL': 'https://www.physionet.org/content/mimic-iv-note/2.2/',
                'Categories': ["Summarization"],
                'Definition': [definition],
                'Reasoning': [],
                'Input_language': ["English"],
                'Output_language': ["English"],
                'Instruction_language': ["English"],
                'Domains': ["Medicine", "Clinical Reports", MODALITY[idx]],
                'Positive Examples': [],
                'Negative Examples': [],
                'Instances': data
            }
            save_path = f'task{idx+75}_mimic_{MODALITY[idx]}_summarization.json'.lower()
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(res_json_dict, f, ensure_ascii=False, indent=4)

def process_modality_and_anatomy_summaries(folder_path, filename):
    """
    Process data to generate summaries based on both modality and anatomy, saving them as JSON files.
    """
    res_data = [[[] for _ in range(len(ANATOMY))] for _ in range(2)]
    with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as fp:
        csv_dict_reader = csv.DictReader(fp)
        for row in tqdm(csv_dict_reader, desc="Processing modalities and anatomy"):
            text = row['text']
            if all(keyword in text for keyword in ['FINDINGS:', 'IMPRESSION:', 'EXAMINATION:', 'INDICATION:']):
                findings_content, impressions_content = extract_findings_and_impressions(text)
                start_index = text.find("EXAMINATION:") + len("EXAMINATION:")
                end_index = text.find("INDICATION:")
                extracted_text = text[start_index:end_index].strip()
                
                modalities = get_modality(extracted_text)
                if 1 in modalities or 2 in modalities:  # Only process for CT and MRI
                    modality_idx = 0 if 1 in modalities else 1
                    anatomy_idx = get_anatomy(extracted_text)
                    if anatomy_idx != -1:
                        res_data[modality_idx][anatomy_idx].append({
                            'input': findings_content,
                            'output': impressions_content
                        })

    save_modality_anatomy_summaries(res_data)

def save_modality_anatomy_summaries(res_data):
    """
    Save summaries as JSON files based on both modality and anatomy.
    """
    for modality_idx, modality in enumerate(['CT', 'MRI']):
        for anatomy_idx, anatomy in enumerate(ANATOMY):
            definition = f"Summarize the {modality} imaging diagnostics' detailed findings for the {anatomy} into a concise conclusion."
            res_json_dict = {
                'Contributors': 'MIMIC',
                'Source': 'MIMIC-IV',
                'URL': 'https://www.physionet.org/content/mimic-iv-note/2.2/',
                'Categories': ["Summarization"],
                'Definition': [definition],
                'Reasoning': [],
                'Input_language': ["English"],
                'Output_language': ["English"],
                'Instruction_language': ["English"],
                'Domains': ["Medicine", "Clinical Reports", modality, anatomy],
                'Positive Examples': [],
                'Negative Examples': [],
                'Instances': res_data[modality_idx][anatomy_idx]
            }
            save_path = f'task{83 + 6 * modality_idx + anatomy_idx}_mimic_{modality}_{anatomy}_summarization.json'.lower()
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(res_json_dict, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    # Define folder path and filename
    folder_path = 'DATA/MIMIC-IV/mimic-iv-note/2.2/note'
    filename = 'radiology.csv'
    
    # Process modality-based and modality-anatomy-based summaries
    process_modality_summaries(folder_path, filename)
    process_modality_and_anatomy_summaries(folder_path, filename)
