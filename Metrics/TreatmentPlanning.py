# treatment_planning.py
# Evaluation script for treatment planning tasks

import argparse
import pandas as pd
from utils import ensure_output_dir

def treatment_planning_accuracy(output, gt):
    """
    Calculate accuracy for treatment planning task
    
    Args:
        output: Model output string
        gt: Ground truth string
        
    Returns:
        1 if correct, 0 otherwise
    """
    output_normalized = output.lower()
    gt_normalized = gt.replace('.', '').replace("The treatment planning is:", "").strip().lower()
    
    return 1 if gt_normalized in output_normalized else 0

def evaluate_treatment_planning(file_paths, task_ids=['task131'], 
                              output_file='USE2_results/Treatment_planning_metrics.csv'):
    """
    Evaluate treatment planning tasks from multiple CSV files
    
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
                    
                response = sample['output'].lower()
                gt_answer = sample['GT'].replace('.', '').replace("The treatment planning is:", "").strip().lower()
                
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
    parser = argparse.ArgumentParser(description="Evaluate treatment planning tasks")
    parser.add_argument("--files", nargs="+", help="List of CSV files to evaluate")
    parser.add_argument("--task_ids", nargs="+", default=['task131'], help="Task IDs to evaluate")
    parser.add_argument("--output", type=str, default="USE2_results/Treatment_planning_metrics.csv", 
                        help="Output file path")
    
    args = parser.parse_args()
    
    if args.files:
        evaluate_treatment_planning(args.files, args.task_ids, args.output)