# fact_verification.py
# Evaluation script for fact verification tasks

import argparse
import pandas as pd
from utils import csv_to_list, cal_similarity, ensure_output_dir

def evaluate_fact_verification(file_paths, task_ids_acc=['task12', 'task16'], 
                             task_ids_bleu=['task100'],
                             output_file='USE2_results/Fact_verification_metrics.csv'):
    """
    Evaluate fact verification tasks using accuracy for some tasks and BLEU/ROUGE for others
    
    Args:
        file_paths: List of CSV file paths to evaluate
        task_ids_acc: List of task IDs to evaluate using accuracy
        task_ids_bleu: List of task IDs to evaluate using BLEU/ROUGE
        output_file: Path to save the results
        
    Returns:
        Tuple of dictionaries with results for each model and task (ACC_result_dict, BLEU_result_dict)
    """
    ensure_output_dir()
    ACC_result_dict, BLEU_result_dict = {}, {}
    
    for filepath in file_paths:
        model_name = filepath.replace('.csv', '').split('_')[-1]
        ACC_result_dict[model_name] = {}
        
        data_list = csv_to_list(filepath)
        
        # Evaluate accuracy tasks
        for cur_task_id in task_ids_acc:
            total_num, acc_num = 0, 0
            
            for item in data_list:
                if item['task_id'] == cur_task_id:
                    ground_truth = item['GT'].replace('.', '').strip().lower()
                    model_output = item['output'].split('\n')[0].strip().lower()
                    
                    if ground_truth in model_output:
                        acc_num += 1
                    total_num += 1
                    
            print(model_name, cur_task_id, total_num)
            
            # Calculate accuracy percentage for this task
            ACC_result_dict[model_name][cur_task_id] = 100 * acc_num / total_num if total_num > 0 else 0
        
        # Evaluate BLEU/ROUGE tasks
        BLEU_result_dict[model_name] = {}
        
        for cur_task_id in task_ids_bleu:
            bleu_sum, rouge_sum, total_num = 0, 0, 0
            
            for item in data_list:
                if item['task_id'] == cur_task_id:
                    ground_truth = item['GT'].replace('.', '').strip().lower()
                    model_output = item['output'].strip().lower()
                    total_num += 1
                    
                    if len(model_output) != 0:
                        bleu_score = cal_similarity(model_output, ground_truth, lang='en', metric='BLEU')
                        rouge_score = cal_similarity(model_output, ground_truth, lang='en', metric='Rouge')
                        bleu_sum += bleu_score
                        rouge_sum += rouge_score
                    else:
                        print("Empty output detected!")
                        
            print(model_name, cur_task_id, total_num)
            
            # Calculate average BLEU and ROUGE scores as percentages
            BLEU_result_dict[model_name][cur_task_id] = (
                100 * bleu_sum / total_num if total_num > 0 else 0, 
                100 * rouge_sum / total_num if total_num > 0 else 0
            )
    
    # Write results to CSV
    with open(output_file, 'w') as fp:
        fp.write('Model Name,' + ','.join(task_ids_acc) + ',' + ','.join(task_ids_bleu) + '\n')
        
        for model_name in ACC_result_dict.keys():
            fp.write(model_name + ',')
            
            # Write accuracy results
            for task_id in task_ids_acc:
                fp.write(f"{ACC_result_dict[model_name][task_id]:.2f},")
            
            # Write BLEU/ROUGE results
            for task_id in task_ids_bleu:
                fp.write(f"{BLEU_result_dict[model_name][task_id][0]:.2f}/{BLEU_result_dict[model_name][task_id][1]:.2f},")
                
            fp.write("\n")
    
    return (ACC_result_dict, BLEU_result_dict)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate fact verification tasks")
    parser.add_argument("--files", nargs="+", help="List of CSV files to evaluate")
    parser.add_argument("--acc_task_ids", nargs="+", default=['task12', 'task16'], 
                       help="Task IDs to evaluate using accuracy")
    parser.add_argument("--bleu_task_ids", nargs="+", default=['task100'], 
                       help="Task IDs to evaluate using BLEU/ROUGE")
    parser.add_argument("--output", type=str, default="USE2_results/Fact_verification_metrics.csv", 
                       help="Output file path")
    
    args = parser.parse_args()
    
    if args.files:
        evaluate_fact_verification(args.files, args.acc_task_ids, args.bleu_task_ids, args.output)
