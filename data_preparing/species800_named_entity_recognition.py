"""
@Description :   Generate JSON file for SuperInstruction Named Entity Recognition task.
@Author      :   Henrychur
"""

import json
from tqdm import tqdm

def reconstruct_sentence(words):
    """
    Reconstructs a sentence from a list of words, handling punctuation spacing.
    
    Parameters:
    - words: List of words and punctuation marks.
    
    Returns:
    - A reconstructed sentence as a single string.
    """
    sentence = words[0] if words else ''
    for word in words[1:]:
        if word in {'.', ',', ';', ':', '?', '!'}:
            sentence += word
        else:
            sentence += ' ' + word
    return sentence

def extract_entities_from_iob(file_path):
    """
    Extracts sentences and entities from an IOB formatted file.

    Parameters:
    - file_path: Path to the IOB file.

    Returns:
    - A list of dictionaries, each containing 'sentence' and 'entities'.
    """
    total_data = []
    words, entities, current_entity = [], [], []
    entity_type = None

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                word, tag = line.split()
                tag_type = tag[0]
                words.append(word)
                
                if tag_type == 'B':  # Start of a new entity
                    if current_entity:
                        entities.append((' '.join(current_entity), entity_type))
                    current_entity = [word]
                    entity_type = tag[2:]
                elif tag_type == 'I' and current_entity:  # Inside entity
                    current_entity.append(word)
                elif tag_type == 'O' and current_entity:  # End of entity
                    entities.append((' '.join(current_entity), entity_type))
                    current_entity = []
            else:  # Sentence boundary
                if current_entity:
                    entities.append((' '.join(current_entity), entity_type))
                if words:
                    total_data.append({
                        'sentence': reconstruct_sentence(words),
                        'entities': entities
                    })
                words, entities, current_entity = [], [], []

    # Append last sentence if file does not end with newline
    if words:
        if current_entity:
            entities.append((' '.join(current_entity), entity_type))
        total_data.append({
            'sentence': reconstruct_sentence(words),
            'entities': entities
        })

    return total_data

def generate_ner_data(input_file, output_file):
    """
    Generates NER task data and saves it to a JSON file.

    Parameters:
    - input_file: Path to the IOB formatted input file.
    - output_file: Path to the output JSON file.
    """
    DEFINITION = (
        "Given a sentence, recognize the names of organisms. "
        "There might be multiple correct answers. If none exist, "
        'output "There is no related entity.".'
    )
    
    result_dict = {
        'Contributors': 'Species-800',
        'Source': 'Species-800',
        'URL': 'https://huggingface.co/datasets/species_800',
        'Categories': ["Named Entity Recognition"],
        'Definition': [DEFINITION],
        'Reasoning': [],
        "Input_language": ["English"],
        "Output_language": ["English"],
        "Instruction_language": ["English"],
        "Domains": ["Public Health", "Healthcare"],
        "Positive Examples": [],
        "Negative Examples": [],
        "Instances": []
    }
    
    total_data = extract_entities_from_iob(input_file)
    
    for item in tqdm(total_data, desc="Processing sentences"):
        sentence = item['sentence']
        entities = item['entities']
        
        if entities:
            output_text = ', '.join([entity for entity, _ in entities]) + '.'
        else:
            output_text = "There is no related entity."
        
        result_dict["Instances"].append({
            "input": sentence,
            "output": output_text
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    input_file = 'test.tsv'
    output_file = 'species800_named_entity_recognition.json'
    
    generate_ner_data(input_file, output_file)
