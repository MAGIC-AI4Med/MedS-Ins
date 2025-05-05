# clinical_outcome.py
# Evaluation script for clinical outcome prediction tasks

import argparse
import pandas as pd
from utils import ensure_output_dir

def clinical_outcome_accuracy(output, gt):
    """
    Calculate accuracy for clinical outcome prediction tasks
    
    Args:
        output: Model output string
        gt: Ground truth string
        
    Returns:
        1 if correct, 0 otherwise
    """
    output_normalized = str(output).lower().strip()
    gt_normalized = str(gt).lower().strip()
    
    if gt_normalized == 'true':
        return 1 if 'true' in output_normalized and 'false' not in output_normalized else 0
    elif gt_normalized == 'false':
        return 1 if 'false' in output_normalized and 'true' not in output_normalized else 0
    else:
        return 0

def evaluate_clinical_outcome(file_paths, task_ids=['task117', 'task118', 'task119'],
                           output_file='USE2_results/Clinical_outcome_prediction_metrics.csv'):
    """
    Evaluate clinical outcome prediction tasks from multiple CSV files
    
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
                gt_answer = str(sample['GT']).lower().strip()
                
                if gt_answer == 'true':
                    if 'true' in response and 'false' not in response:
                        right_cnt += 1
                elif gt_answer == 'false':
                    if 'false' in response and 'true' not in response:
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
    parser = argparse.ArgumentParser(description="Evaluate clinical outcome prediction tasks")
    parser.add_argument("--files", nargs="+", help="List of CSV files to evaluate")
    parser.add_argument("--task_ids", nargs="+", default=['task117', 'task118', 'task119'], 
                       help="Task IDs to evaluate")
    parser.add_argument("--output", type=str, default="USE2_results/Clinical_outcome_prediction_metrics.csv", 
                       help="Output file path")
    
    args = parser.parse_args()
    
    if args.files:
        evaluate_clinical_outcome(args.files, args.task_ids, args.output)
