# diagnosis.py
# Evaluation script for diagnosis tasks

import argparse
import pandas as pd
from utils import csv_to_list, ensure_output_dir

def diagnosis_accuracy(output, gt):
    """
    Calculate accuracy for diagnosis task
    
    Args:
        output: Model output string
        gt: Ground truth string
        
    Returns:
        1 if correct, 0 otherwise
    """
    gt_normalized = gt.replace('.', '').replace("The diagnosis result is", "").strip().lower()
    output_normalized = output.lower()
    
    return 1 if gt_normalized in output_normalized else 0

def evaluate_diagnosis(file_paths, output_file='USE2_results/Diagnosis_metrics.csv'):
    """
    Evaluate diagnosis tasks from multiple CSV files
    
    Args:
        file_paths: List of CSV file paths to evaluate
        output_file: Path to save the results
        
    Returns:
        Dictionary with results for each model
    """
    ensure_output_dir()
    results = {}
    
    for filepath in file_paths:
        total_num, acc_num = 0, 0
        model_name = filepath.replace('.csv', '').split('_')[-1]
        
        data_list = csv_to_list(filepath)
        for item in data_list:
            ground_truth = item['GT']
            model_output = item['output']
            
            if diagnosis_accuracy(model_output, ground_truth):
                acc_num += 1
            total_num += 1
        
        accuracy = 100 * acc_num / total_num if total_num > 0 else 0
        results[model_name] = accuracy
        print(f'{model_name}:\t{acc_num}\t{total_num}\t{accuracy:.2f}%')
    
    # Write results to CSV
    with open(output_file, 'w') as fp:
        fp.write('Model Name,Accuracy\n')
        for model_name, accuracy in results.items():
            fp.write(f'{model_name},{accuracy:.2f}\n')
    
    return results

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate diagnosis tasks")
    parser.add_argument("--files", nargs="+", help="List of CSV files to evaluate")
    parser.add_argument("--output", type=str, default="USE2_results/Diagnosis_metrics.csv", 
                        help="Output file path")
    
    args = parser.parse_args()
    
    if args.files:
        evaluate_diagnosis(args.files, args.output)
