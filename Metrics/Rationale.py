# reasoning.py
# Evaluation script for reasoning tasks in different languages

import argparse
import pandas as pd
from utils import cal_similarity, ensure_output_dir

def evaluate_reasoning(file_paths, task_ids=["task51", "task52", "task53", "task54", "task55", "task56"],
                     output_file='USE2_results/Reasoning_metrics.csv'):
    """
    Evaluate reasoning tasks using BLEU and ROUGE metrics across different languages
    
    Args:
        file_paths: List of CSV file paths to evaluate
        task_ids: List of task IDs to evaluate
        output_file: Path to save the results
        
    Returns:
        Dictionary with results for each model and task
    """
    # Map task IDs to languages
    TASKID2LANG = {
        'task51': 'zh',  # Chinese
        'task52': 'en',  # English
        'task53': 'fra', # French
        'task54': 'jp',  # Japanese
        'task55': 'ru',  # Russian
        'task56': 'spa', # Spanish
    }
    
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
                    
                response = sample['output'].strip()
                gt_answer = sample['GT'].strip()
                
                if (not isinstance(response, str)) or (not isinstance(gt_answer, str)):
                    continue
                
                if len(response) == 0 or len(gt_answer) == 0:
                    continue
                    
                # Get the language for this task
                lang = TASKID2LANG.get(cur_task_id, 'en')
                
                # Calculate BLEU and ROUGE scores with the appropriate language
                score_bleu1 = cal_similarity(response, gt_answer, lang=lang, metric='BLEU')
                score_rouge1 = cal_similarity(response, gt_answer, lang=lang, metric='Rouge')
                
                bleu1_sum += score_bleu1
                rouge1_sum += score_rouge1
                cnt += 1
                
            # Calculate average BLEU and ROUGE scores as percentages
            if cnt > 0:
                result_dict[model_name][cur_task_id] = (
                    100 * bleu1_sum / cnt, 
                    100 * rouge1_sum / cnt
                )
            else:
                result_dict[model_name][cur_task_id] = (0, 0)
    
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
    parser = argparse.ArgumentParser(description="Evaluate reasoning tasks")
    parser.add_argument("--files", nargs="+", help="List of CSV files to evaluate")
    parser.add_argument("--task_ids", nargs="+", 
                      default=["task51", "task52", "task53", "task54", "task55", "task56"], 
                      help="Task IDs to evaluate")
    parser.add_argument("--output", type=str, default="USE2_results/Reasoning_metrics.csv", 
                      help="Output file path")
    
    args = parser.parse_args()
    
    if args.files:
        evaluate_reasoning(args.files, args.task_ids, args.output)
