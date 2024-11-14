"""
@Description :   MIMIC-IV-based prompt classification tasks for imaging modality and anatomy classification.
@Author      :   Henrychur
@Time        :   2024/03/28 16:13:26
"""

import os
import csv
import tqdm
import json

# Constants
MODALITY = ['X-ray', 'CT', 'MRI', 'Ultrasound', 'Angiography', 'Mammogram', 'Pathology', 'Fluoroscopy']
ANATOMY = ['Chest', 'Head_and_neck', 'Brain', 'Abdomen', 'Pelvis', 'Spine']
MAX_SAMPLES = 10000

def get_modality(extracted_text):
    """Identifies the imaging modalities in the given text."""
    res_list = []
    splited_text = extracted_text.split()
    lower_text = extracted_text.lower()
    if 'x-ray' in lower_text:
        res_list.append(0)
    if 'CT' in splited_text or 'CTA' in splited_text:
        res_list.append(1)
    if 'MRI' in splited_text or 'MR' in splited_text or 'MRA' in splited_text:
        res_list.append(2)
    if 'ultrasound' in lower_text:
        res_list.append(3)
    if 'angiography' in lower_text:
        res_list.append(4)
    if 'mammogram' in lower_text:
        res_list.append(5)
    if 'pathology' in lower_text:
        res_list.append(6)
    if 'fluoroscopy' in lower_text:
        res_list.append(7)
    return res_list

def get_anatomy(extracted_text):
    """Identifies the anatomical location based on the given text."""
    lower_text = extracted_text.lower()
    if 'chest' in lower_text:
        return 0
    if 'head' in lower_text or 'neck' in lower_text:
        return 1
    if 'brain' in lower_text:
        return 2
    if 'abdomen' in lower_text:
        return 3
    if 'pelvis' in lower_text:
        return 4
    if 'spine' in lower_text:
        return 5
    return -1

def prompt_modality_classification_task():
    """Classifies modality based on extracted text from radiology reports."""
    definition = 'Identify the imaging modality used for the diagnosis based on diagnostic findings.'
    counter = [0] * len(MODALITY)
    folder_path = 'DATA/MIMIC-IV/mimic-iv-note/2.2/note'
    filename = 'radiology.csv'
    res_data = []
    
    with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as fp:
        csv_dict_reader = csv.DictReader(fp)
        for row in tqdm.tqdm(csv_dict_reader):
            text = row['text']
            if all(keyword in text for keyword in ['FINDINGS:', 'IMPRESSION:', 'EXAMINATION:', 'INDICATION:']):
                findings_content, impressions_content, extracted_text = extract_relevant_texts(text)
                modalities = get_modality(extracted_text)
                
                if modalities:
                    if add_classification_instance(res_data, counter, modalities, findings_content, impressions_content):
                        counter = update_counter(counter, modalities)
    
    save_task_data('task95_mimic_modality_classification.json', res_data, definition)

def prompt_anatomy_classification_task():
    """Classifies anatomical location based on extracted text from radiology reports."""
    definition = 'Identify the anatomical location of the diagnosis based on diagnostic findings.'
    counter = [0] * len(ANATOMY)
    folder_path = 'DATA/MIMIC-IV/mimic-iv-note/2.2/note'
    filename = 'radiology.csv'
    res_data = []

    with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as fp:
        csv_dict_reader = csv.DictReader(fp)
        for row in tqdm.tqdm(csv_dict_reader):
            text = row['text']
            if all(keyword in text for keyword in ['FINDINGS:', 'IMPRESSION:', 'EXAMINATION:', 'INDICATION:']):
                findings_content, impressions_content, extracted_text = extract_relevant_texts(text)
                anatomy = get_anatomy(extracted_text)
                
                if anatomy != -1 and counter[anatomy] < MAX_SAMPLES:
                    res_data.append({
                        'input': f'{findings_content}\n{impressions_content}',
                        'output': f'The anatomical location of the diagnosis is {ANATOMY[anatomy]}.'
                    })
                    counter[anatomy] += 1
    
    save_task_data('task96_mimic_anatomy_classification.json', res_data, definition)

def extract_relevant_texts(text):
    """Extracts findings, impressions, and examination texts from the given report text."""
    findings_index = text.find("FINDINGS:")
    impressions_index = text.find("IMPRESSION:")
    findings_content = text[findings_index + len("FINDINGS: "):impressions_index].strip()
    impressions_content = text[impressions_index + len("IMPRESSION: "):].strip()

    start_index = text.find("EXAMINATION:") + len("EXAMINATION:")
    end_index = text.find("INDICATION:")
    extracted_text = text[start_index:end_index].strip()
    
    return findings_content, impressions_content, extracted_text

def add_classification_instance(res_data, counter, modalities, findings_content, impressions_content):
    """Adds classification instance to results if sample count limits allow."""
    if len(modalities) == 1 and counter[modalities[0]] < MAX_SAMPLES:
        res_data.append({
            'input': f'{findings_content}\n{impressions_content}',
            'output': f'The imaging modality used for the diagnosis is {MODALITY[modalities[0]]}.'
        })
        return True
    elif len(modalities) > 1:
        for idx in modalities:
            if counter[idx] >= MAX_SAMPLES:
                return False
        res_data.append({
            'input': f'{findings_content}\n{impressions_content}',
            'output': f'The imaging modalities used for the diagnosis are ' + ', '.join([MODALITY[i] for i in modalities]) + '.'
        })
        return True
    return False

def update_counter(counter, modalities):
    """Updates sample counter for each identified modality."""
    for idx in modalities:
        counter[idx] += 1
    return counter

def save_task_data(filename, res_data, definition):
    """Saves classification task data to a JSON file."""
    res_json_dict = {
        'Contributors': 'MIMIC',
        'Source': 'MIMIC-IV',
        'URL': 'https://www.physionet.org/content/mimic-iv-note/2.2/',
        'Categories': ["Classification"],
        'Definition': [definition],
        'Reasoning': [],
        'Input_language': ["English"],
        'Output_language': ["English"],
        'Instruction_language': ["English"],
        'Domains': ['Medicine', "Clinical Reports"],
        'Positive Examples': [],
        'Negative Examples': [],
        'Instances': res_data
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(json.dumps(res_json_dict, ensure_ascii=False, indent=4) + '\n')

if __name__ == '__main__':
    prompt_modality_classification_task()
    prompt_anatomy_classification_task()
