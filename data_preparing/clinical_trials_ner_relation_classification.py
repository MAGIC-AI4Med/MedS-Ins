import json
import os
import tqdm

# 初始化结果字典
result_dict = {
    "Contributors": "Chaoyi Wu",
    "Source": "clinical-trials-gov-data",
    "URL": "https://github.com/uw-bionlp/clinical-trials-gov-data",
    "Categories": ["Named Entity Recognition", "Word Relation Classification", "Entity Relation Classification"],
    "Definition": [
        "Identifying cohorts of patients based on eligibility criteria such as medical conditions, procedures, and medication use is critical to recruitment for clinical trials. "
        "Such criteria are often most naturally described in free-text, using language familiar to clinicians and researchers. In order to identify potential participants at scale, "
        "these criteria must first be translated into queries on clinical databases, which can be labor-intensive and error-prone. Thus, in the following task, I hope you can help "
        "me structure the report criteria. In detail, you have to recognize the vital medical entities from the free-texts, together with their relationships. For entities, you "
        "need to pick a type for them from a predefined list. And for relationships, you need to pick from a specified list of relations. I hope you organize your answer as JSON format, "
        "indexed by line number. For each line, output two elements: 'Entities' (a list of (entity, entity_type) tuples) and 'Relations' (a list of (start_entity, end_entity, relation_type) "
        "tuples). All elements should follow their order of appearance in the original free-text."
    ],
    "Reasoning": [],
    "Input_language": ["English"],
    "Output_language": ["English"],
    "Instruction_language": ["English"],
    "Domains": ["Medicine", "Clinical Trials", "Eligibility Criteria"],
    "Positive Examples": [],
    "Negative Examples": [],
    "Instances": []
}

def find_txt_files(directory):
    """
    Recursively search for all .txt files in the specified directory and return a dictionary.
    Key: filename without extension, Value: full file path.
    """
    txt_files = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_name_without_extension = os.path.splitext(file)[0]
                txt_files[file_name_without_extension] = os.path.join(root, file)
    return txt_files

def reorganize_by_orig_id(dicts_list):
    """
    Organize a list of dictionaries by the numeric part of the 'orig_id' key (after the underscore).
    """
    organized_dict = {}
    for d in dicts_list:
        _, id_part = d['orig_id'].split('_')
        if id_part not in organized_dict:
            organized_dict[id_part] = []
        organized_dict[id_part].append(d)
    return organized_dict

def read_file_to_string(file_path):
    """
    Read the content of a text file and return it as a single string.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return ""

def fill_missing_lines_and_sort(orig_dict):
    """
    Fill missing lines (LineX) in the dictionary and return a sorted version.
    """
    max_line_number = max(int(key.replace('Line', '')) for key in orig_dict.keys())
    for i in range(max_line_number + 1):
        line_key = f'Line{i}'
        if line_key not in orig_dict:
            orig_dict[line_key] = {"Entities": [], "Relations": []}
    return dict(sorted(orig_dict.items(), key=lambda item: int(item[0].replace('Line', ''))))

# 定义目录路径
TEXT_DIR = "Dataset/Medical/clinical-trials-gov-data/data/docs/brat/"
ANNOTATION_FILE = 'Dataset/Medical/clinical-trials-gov-data/data/ner/spert/train.json'

TXT_DIRE = find_txt_files(TEXT_DIR)

with open(ANNOTATION_FILE, 'r') as f:
    ANNOTATION_DICT = reorganize_by_orig_id(json.load(f))

for item in tqdm.tqdm(ANNOTATION_DICT.keys()):
    trial_criterials = read_file_to_string(TXT_DIRE.get(item, ""))
    annotation_label = ANNOTATION_DICT[item]
    
    input_ss = trial_criterials
    output_dict = {}

    for elem in annotation_label:
        line_number = elem['orig_id'].split('_')[0]
        entities, relations = [], []

        for entity in elem.get('entities', []):
            token = ' '.join(elem['tokens'][entity['start']:entity['end']])
            entities.append({token: entity['type']})

        for relation in elem.get('relations', []):
            relations.append((entities[relation["head"]], entities[relation["tail"]], relation["type"]))
        
        output_dict[f'Line{line_number}'] = {
            'Entities': entities,
            'Relations': relations
        }

    output_dict = fill_missing_lines_and_sort(output_dict)
    output_ss = json.dumps(output_dict, ensure_ascii=False, indent=4)

    result_dict["Instances"].append({
        "input": input_ss,
        "output": output_ss
    })

with open("task77_clinical_trials_gov_data_named_entity_recognition.json", 'w', encoding='utf-8') as f:
    json.dump(result_dict, f, ensure_ascii=False, indent=4)
