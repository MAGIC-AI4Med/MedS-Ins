# information_extraction.py
# Evaluation script for information extraction tasks

import argparse
import pandas as pd
from utils import ensure_output_dir

def information_extraction_accuracy(output, gt):
    """
    Calculate accuracy for information extraction task
    
    Args:
        output: Model output string
        gt: Ground truth string
        
    Returns:
        1 if correct, 0 otherwise
    """
    output_normalized = str(output).lower().strip().replace('"', '').replace("'",'').replace("\n", '')
    gt_normalized = str(gt).lower().strip().replace('"', '').replace("'",'').replace("\n", '')
    
    return 1 if gt_normalized in output_normalized else 0

def evaluate_information_extraction(file_paths, task_ids=['task1', 'task2', 'task3', 'task29', 'task74'],
                                  output_file='USE2_results/information_extraction_metrics.csv'):
    """
    Evaluate information extraction tasks from multiple CSV files
    
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
                    
                response = str(sample['output']).lower().strip().replace('"', '').replace("'",'').replace("\n", '')
                gt_answer = str(sample['GT']).lower().strip().replace('"', '').replace("'",'').replace("\n", '')
                
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

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate information extraction tasks")
    parser.add_argument("--files", nargs="+", help="List of CSV files to evaluate")
    parser.add_argument("--task_ids", nargs="+", default=['task1', 'task2', 'task3', 'task29', 'task74'], 
                      help="Task IDs to evaluate")
    parser.add_argument("--output", type=str, default="USE2_results/information_extraction_metrics.csv", 
                      help="Output file path")
    
    args = parser.parse_args()
    
    if args.files:
        evaluate_information_extraction(args.files, args.task_ids, args.output)
