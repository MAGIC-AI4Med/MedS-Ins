# hard_ner.py
# Evaluation script for Hard Named Entity Recognition tasks (multiple entities in GT)

import argparse
import pandas as pd
from utils import csv_to_list, ensure_output_dir

def parse_entities(text):
    """
    Parse entity text into a set of entities
    
    Args:
        text: Text containing entities
        
    Returns:
        Set of entity strings
    """
    text = text.replace('.', '').strip().lower()
    
    if text == "there is no related enetity":
        return set()
    else:
        # Split by comma and remove the last period if exists
        return set(text[:-1].split(", "))

def hard_ner_f1(output, gt):
    """
    Calculate F1 score for hard NER task (multiple entities)
    
    Args:
        output: Model output string
        gt: Ground truth string
        
    Returns:
        F1 score
    """
    if not isinstance(output, str):
        return 0
        
    gt_entities = parse_entities(gt)
    output_entities = parse_entities(output)
    
    # Calculate TP, FP, FN
    tp = len(gt_entities.intersection(output_entities))
    fp = len(output_entities - gt_entities)
    fn = len(gt_entities - output_entities)
    
    # Calculate precision, recall, and F1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1_score

def evaluate_hard_ner(file_paths, task_ids=['task125', 'task126', 'task127', 'task128'],
                    output_file='USE2_results/Hard_NER_metrics.csv'):
    """
    Evaluate Hard NER tasks from multiple CSV files
    
    Args:
        file_paths: List of CSV file paths to evaluate
        task_ids: List of task IDs to evaluate
        output_file: Path to save the results
        
    Returns:
        Dictionary with results for each model and task
    """
    ensure_output_dir()
    result_dict = {}
    
    for filepath in file_paths:
        model_name = filepath.replace('.csv', '').split('_')[-1]
        result_dict[model_name] = {}
        
        data_list = csv_to_list(filepath)
        
        for cur_task_id in task_ids:
            total_tp = total_fp = total_fn = 0
            
            for item in data_list:
                if item['task_id'] == cur_task_id:
                    if not isinstance(item['output'], str):
                        continue
                        
                    gt = item['GT'].replace('.', '').strip().lower()
                    output = item['output'].replace('.', '').strip().lower()
                    
                    if gt == "there is no related enetity":
                        gt_entities = set()
                    else:
                        gt_entities = set(gt[:-1].split(", "))
                    
                    if output == "there is no related enetity":
                        output_entities = set()
                    else:
                        output_entities = set(output[:-1].split(", "))
                    
                    # Calculate TP, FP, FN
                    tp = len(gt_entities.intersection(output_entities))
                    fp = len(output_entities - gt_entities)
                    fn = len(gt_entities - output_entities)
                    
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn
            
            # Calculate precision, recall, and F1 score
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Store F1 score as percentage
            result_dict[model_name][cur_task_id] = 100 * f1_score
    
    # Write results to CSV
    with open(output_file, 'w') as fp:
        fp.write('Model Name,' + ','.join(task_ids) + ', Avg.\n')
        
        for model_name in result_dict.keys():
            fp.write(model_name + ',')
            
            for task_id in task_ids:
                fp.write(f"{result_dict[model_name][task_id]:.2f},")
                
            # Calculate and write average across all tasks
            avg = sum(result_dict[model_name].values()) / len(result_dict[model_name].values())
            fp.write(f"{avg:.2f}\n")
    
    return result_dict

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate Hard NER tasks")
    parser.add_argument("--files", nargs="+", help="List of CSV files to evaluate")
    parser.add_argument("--task_ids", nargs="+", default=['task125', 'task126', 'task127', 'task128'], 
                        help="Task IDs to evaluate")
    parser.add_argument("--output", type=str, default="USE2_results/Hard_NER_metrics.csv", 
                        help="Output file path")
    
    args = parser.parse_args()
    
    if args.files:
        evaluate_hard_ner(args.files, args.task_ids, args.output)
