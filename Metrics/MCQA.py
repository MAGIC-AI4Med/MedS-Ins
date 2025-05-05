# mcqa.py
# Evaluation script for Multiple Choice Question Answering (MCQA) tasks

import argparse
import pandas as pd
from utils import csv_to_list, write_results_to_csv

def mcqa_accuracy(pred, gt, task_id=None):
    """
    Calculate accuracy for MCQA task
    
    Args:
        pred: Model prediction string
        gt: Ground truth string
        task_id: Task ID to handle special cases
        
    Returns:
        1 if correct answer, 0 otherwise
    """
    # Normalize and extract answer
    pred = str(pred).lower().replace('the right answer is', '').strip()
    gt = str(gt).lower().replace('the right answer is', '').strip()
    
    # Special handling for yes/no questions
    if task_id in ['task123']:
        if pred[0] in 'yn' and gt[0] in 'yn':
            return 1 if pred[0] == gt[0] else 0
    else:
        # Standard MCQA (multiple choice A-D)
        if pred[0] in 'abcd' and gt[0] in 'abcd':
            return 1 if pred[0] == gt[0] else 0
    
    # If we couldn't parse answers in expected format
    return 0

def evaluate_mcqa(file_paths, task_ids=['task4', 'task5', 'task6', 'task8', 'task9', 
                                     'task57', 'task58', 'task59', 'task60', 'task61', 
                                     'task122', 'task123', 'task129']):
    """
    Evaluate MCQA tasks from multiple CSV files
    
    Args:
        file_paths: List of CSV file paths to evaluate
        task_ids: List of task IDs to evaluate
        
    Returns:
        Dictionary with results for each model and task
    """
    result_dict = {}
    
    for idx, filepath in enumerate(file_paths):
        # Extract model name from filepath
        model_name = filepath.replace('.csv', '').split('_')[-1]
        result_dict[model_name] = {}
        
        # Read CSV file
        df = pd.read_csv(filepath)
        
        for cur_task_id in task_ids:
            cnt, right_cnt = 0, 0
            
            for index in range(len(df)):
                sample = df.iloc[index]
                task_id = sample['task_id']
                
                if task_id != cur_task_id:
                    continue
                    
                response = str(sample['output'])
                gt_answer = str(sample['GT'])
                
                # Check if answer is correct
                is_correct = mcqa_accuracy(response, gt_answer, task_id)
                if is_correct:
                    right_cnt += 1
                cnt += 1
            
            if cnt > 0:
                # Print detailed results for debugging
                print(f"{model_name}, {cur_task_id}, {right_cnt}/{cnt}, {100*right_cnt/cnt:.2f}%")
                
                # Calculate accuracy percentage for this task
                result_dict[model_name][cur_task_id] = 100 * right_cnt / cnt
            else:
                print(f"{model_name}, {cur_task_id}, 0/0, 0.00%")
                result_dict[model_name][cur_task_id] = 0
    
    # Write results to CSV
    write_results_to_csv(result_dict, 'MCQA_metrics.csv', task_ids, 'accuracy')
    
    return result_dict

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate MCQA tasks")
    parser.add_argument("--files", nargs="+", help="List of CSV files to evaluate")
    parser.add_argument("--task_ids", nargs="+", help="List of task IDs to evaluate")
    
    args = parser.parse_args()
    
    if args.files:
        task_ids = args.task_ids if args.task_ids else [
            'task4', 'task5', 'task6', 'task8', 'task9',
            'task57', 'task58', 'task59', 'task60', 'task61',
            'task122', 'task123', 'task129'
        ]
        evaluate_mcqa(args.files, task_ids)
    else:
        # Test with a single file
        test_file = "./test_data/MCQA_sample.csv"
        print(f"Testing with sample file: {test_file}")
        
        # Test with a small set of tasks
        test_tasks = ['task4', 'task5']
        evaluate_mcqa([test_file], test_tasks)