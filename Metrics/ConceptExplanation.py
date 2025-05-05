# explanation.py
# Evaluation script for explanation tasks

import argparse
import pandas as pd
from utils import cal_similarity, ensure_output_dir

def evaluate_explanation(file_paths, task_ids=['task18', 'task46', 'task50'],
                       output_file='USE2_results/Explanation_metrics.csv'):
    """
    Evaluate explanation tasks using BLEU and ROUGE metrics
    
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
        
        df = pd.read_csv(filepath, on_bad_lines='warn')
        
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
                
                if (not isinstance(response, str)) or (not isinstance(gt_answer, str)):
                    continue
                    
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
    parser = argparse.ArgumentParser(description="Evaluate explanation tasks")
    parser.add_argument("--files", nargs="+", help="List of CSV files to evaluate")
    parser.add_argument("--task_ids", nargs="+", default=['task18', 'task46', 'task50'], 
                        help="Task IDs to evaluate")
    parser.add_argument("--output", type=str, default="USE2_results/Explanation_metrics.csv", 
                        help="Output file path")
    
    args = parser.parse_args()
    
    if args.files:
        evaluate_explanation(args.files, args.task_ids, args.output)
