"""
@Description :   Generate JSON files for SuperInstruction dataset from igakuqa.
                 Since MMedBench contain igakuqa, so we generate data from MMedbench-JA part.
@Author      :   Henrychur 
@Time        :   2024/03/22 15:46:55
"""

import json
from tqdm import tqdm

DEFINITIONS = {
    'igakuqa_question_answering': 'Given a question and a list of options, select the correct answer from the options directly.'
}

def generate_igakuqa_data(filepath, output_filename='igakuqa_question_answering.jsonl'):
    """
    Processes IgakuQA data from a JSONL file and generates formatted data for SuperInstruction.

    :param filepath: Path to the input JSONL file containing IgakuQA data.
    :param output_filename: Name of the output JSON file.
    """
    src_data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            src_data.append(json.loads(line.strip()))
    
    res_data = []
    for item in tqdm(src_data, desc="Processing IgakuQA data"):
        question = item['question']
        options = item['options']
        answer_idx = item['answer_idx']

        # Format options as a string with lettered keys and corresponding text
        options_string = '\t'.join([f"{key}: {options[key]}" for key in options.keys()])

        # Prepare the answer based on single or multiple correct answers
        if len(answer_idx) == 1:
            answer = f"{answer_idx[0]}: {options[answer_idx[0]]}"
        else:
            answer = ', '.join([f"{idx}: {options[idx]}" for idx in answer_idx])

        res_data.append({
            'input': f'質問: {question}\nオプション: {options_string}',
            'output': f'正しい答えは {answer} です'
        })
    
    # Create the final JSON structure
    res_json_dict = {
        'Contributors': 'Jungo Kasai',
        'Source': 'IgakuQA',
        'URL': 'https://github.com/jungokasai/IgakuQA',
        'Categories': ["Question Answering"],
        'Definition': [DEFINITIONS['igakuqa_question_answering']],
        'Reasoning': [],
        'Input_language': ["Japanese"],
        'Output_language': ["Japanese"],
        'Instruction_language': ["English"],
        'Domains': ["Public Health"],
        'Positive Examples': [],
        'Negative Examples': [],
        'Instances': res_data
    }

    # Save the data as a JSONL file
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(json.dumps(res_json_dict, ensure_ascii=False, indent=4) + '\n')

def generate_from_multiple_choice():
    """
    Generates JSON files for SuperInstruction from multiple-choice question datasets.
    Currently only supports 'IgakuQA' dataset.
    """
    generate_igakuqa_data('MMedBench/Train/Japanese.jsonl')

if __name__ == '__main__':
    generate_from_multiple_choice()
