# nli.py
# Evaluation script for Natural Language Inference tasks (both NLI and NLI_GEN)

import argparse
import pandas as pd
from utils import csv_to_list, cal_similarity, ensure_output_dir

def nli_accuracy(output, gt):
    """
    Calculate accuracy for NLI classification task
    
    Args:
        output: Model output string
        gt: Ground truth string
        
    Returns:
        1 if correct, 0 otherwise
    """
    output_normalized = str(output).lower().strip()
    gt_normalized = str(gt).replace('.', '').strip().lower()
    
    return 1 if gt_normalized in output_normalized else 0

def evaluate_nli(file_paths, task_ids=['task41'], 
               output_file='USE2_results/NLI_metrics.csv'):
    """
    Evaluate NLI classification tasks from multiple CSV files
    
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
        
        df = pd.read_csv(filepath)
        
        for cur_task_id in task_ids:
            cnt, right_cnt = 0, 0
            
            for index in range(len(df)):
                sample = df.iloc[index]
                task_id = sample['task_id']
                
                if task_id != cur_task_id:
                    continue
                    
                response = str(sample['output']).lower().strip()
                gt_answer = str(sample['GT']).replace('.', '').strip().lower()
                
                if gt_answer in response:
                    right_cnt += 1
                cnt += 1
                
            # Calculate accuracy percentage for this task
            result_dict[model_name][cur_task_id] = 100 * right_cnt / cnt if cnt > 0 else 0
    
    # Write results to CSV
    with open(output_file, 'w') as fp:
        fp.write('Model Name,' + ','.join(task_ids) + ', Avg.\n')
        
        for model_name in result_dict.keys():
            fp.write(model_name + ',')
            
            for task_id in task_ids:
                fp.write(f"{result_dict[model_name][task_id]:.2f},")
                
            # Calculate and write average across all tasks
            avg = sum([result_dict[model_name][task_id] for task_id in task_ids]) / len(task_ids)
            fp.write(f"{avg:.2f}\n")
    
    return result_dict

def evaluate_nli_gen(file_paths, task_ids=['task42'], 
                   output_file='USE2_results/NLI_GEN_metrics.csv'):
    """
    Evaluate NLI generation tasks from multiple CSV files
    
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
        
        df = pd.read_csv(filepath)
        
        for cur_task_id in task_ids:
            cnt = 0
            bleu1_sum, rouge1_sum = 0, 0
            
            for index in range(len(df)):
                sample = df.iloc[index]
                task_id = sample['task_id']
                
                if task_id != cur_task_id:
                    continue
                    
                response = sample['output']
                gt_answer = sample['GT']
                
                # Calculate BLEU and ROUGE scores
                score_bleu1 = cal_similarity(response, gt_answer, lang='en', metric='BLEU')
                score_rouge1 = cal_similarity(response, gt_answer, lang='en', metric='Rouge')
                
                bleu1_sum += score_bleu1
                rouge1_sum += score_rouge1
                cnt += 1
                
            # Calculate average BLEU and ROUGE scores as percentages
            result_dict[model_name][cur_task_id] = (
                100 * bleu1_sum / cnt if cnt > 0 else 0, 
                100 * rouge1_sum / cnt if cnt > 0 else 0
            )
    
    # Write results to CSV
    with open(output_file, 'w') as fp:
        fp.write('Model Name,' + ','.join(task_ids) + ', Avg.\n')
        
        for model_name in result_dict.keys():
            fp.write(model_name + ',')
            
            for task_id in task_ids:
                fp.write(f"{result_dict[model_name][task_id][0]:.2f}/{result_dict[model_name][task_id][1]:.2f},")
                
            # Calculate and write average BLEU/ROUGE across all tasks
            avg_bleu = sum([result_dict[model_name][task_id][0] for task_id in task_ids]) / len(task_ids)
            avg_rouge = sum([result_dict[model_name][task_id][1] for task_id in task_ids]) / len(task_ids)
            fp.write(f"{avg_bleu:.2f}/{avg_rouge:.2f}\n")
    
    return result_dict

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate NLI tasks")
    parser.add_argument("--files", nargs="+", help="List of CSV files to evaluate")
    parser.add_argument("--type", choices=["classification", "generation", "both"], default="both",
                        help="Type of NLI task to evaluate")
    parser.add_argument("--classification_task_ids", nargs="+", default=['task41'], 
                        help="Task IDs for classification")
    parser.add_argument("--generation_task_ids", nargs="+", default=['task42'], 
                        help="Task IDs for generation")
    parser.add_argument("--classification_output", type=str, default="USE2_results/NLI_metrics.csv", 
                        help="Output file path for classification")
    parser.add_argument("--generation_output", type=str, default="USE2_results/NLI_GEN_metrics.csv", 
                        help="Output file path for generation")
    
    args = parser.parse_args()
    
    if args.files:
        if args.type in ["classification", "both"]:
            evaluate_nli(args.files, args.classification_task_ids, args.classification_output)
        
        if args.type in ["generation", "both"]:
            evaluate_nli_gen(args.files, args.generation_task_ids, args.generation_output)