"""
@Description :   Generate JSON files for SuperInstruction tasks: Question Answering and Justification.
@Author      :   Henrychur
"""

import json
import xml.etree.ElementTree as ET
from tqdm import tqdm

def initialize_result_dict(contributors, source, url, categories, definition, domains):
    """
    Initializes the result dictionary with metadata and basic structure.

    Parameters:
    - contributors: The contributors of the dataset.
    - source: The source of the dataset.
    - url: URL for more information.
    - categories: List of categories for the task.
    - definition: Instruction definition for the task.
    - domains: List of domains for the task.

    Returns:
    - A dictionary with metadata and structure for storing instances.
    """
    return {
        "Contributors": contributors,
        "Source": source,
        "URL": url,
        "Categories": categories,
        "Definition": [definition],
        "Reasoning": [],
        "Input_language": ["English"],
        "Output_language": ["English"],
        "Instruction_language": ["English"],
        "Domains": domains,
        "Positive Examples": [],
        "Negative Examples": [],
        "Instances": []
    }

def parse_xml(filepath):
    """
    Parses the XML file and retrieves all records.

    Parameters:
    - filepath: Path to the XML file.

    Returns:
    - A list of XML elements representing records.
    """
    tree = ET.parse(filepath)
    return tree.findall('record')

def generate_question_answering_task(records, output_file):
    """
    Generates question answering task instances and saves them to a JSON file.

    Parameters:
    - records: List of XML elements representing records.
    - output_file: Path to the output JSON file.
    """
    result_dict = initialize_result_dict(
        contributors="EBMSummariserCorpus",
        source="EBMSummariserCorpus",
        url="https://sourceforge.net/projects/ebmsumcorpus/",
        categories=["Question Answering"],
        definition="Given a question, answer it with comprehensive details.",
        domains=["Public Health", "Healthcare"]
    )
    
    for record in tqdm(records, desc="Generating Question Answering Task"):
        question = record.find('question').text
        answer_text_list = [snip.find('sniptext').text for snip in record.find('answer').findall('snip')]
        answer = ' '.join(answer_text_list)

        result_dict["Instances"].append({
            "input": question,
            "output": answer
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)

def generate_justification_task(records, output_file):
    """
    Generates justification task instances and saves them to a JSON file.

    Parameters:
    - records: List of XML elements representing records.
    - output_file: Path to the output JSON file.
    """
    result_dict = initialize_result_dict(
        contributors="EBMSummariserCorpus",
        source="EBMSummariserCorpus",
        url="https://sourceforge.net/projects/ebmsumcorpus/",
        categories=["Question Answering"],
        definition="Provide a justification or evidence to support a given statement.",
        domains=["Public Health", "Healthcare"]
    )
    
    for record in tqdm(records, desc="Generating Justification Task"):
        for snip in record.find('answer').findall('snip'):
            statement = snip.find('sniptext').text
            justifications_list = snip.findall('long')

            for justification in justifications_list:
                justification_text = justification.find('longtext').text
                result_dict["Instances"].append({
                    "input": statement,
                    "output": justification_text
                })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # Define file paths
    input_xml_path = 'ClinicalInquiries.xml'
    output_qa_file = 'EBMS_question_answering.json'
    output_justification_file = 'EBMS_answer_verification.json'
    
    # Parse XML records
    records = parse_xml(input_xml_path)
    
    # Generate tasks
    generate_question_answering_task(records, output_qa_file)
    generate_justification_task(records, output_justification_file)
