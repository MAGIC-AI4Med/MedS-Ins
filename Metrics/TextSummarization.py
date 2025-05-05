# summarization.py
# Evaluation script for text summarization tasks

import argparse
import pandas as pd
from utils import cal_similarity, ensure_output_dir

def evaluate_summarization(file_paths, task_ids=['task78', 'task114', 'task112', 'task110'],
                         task_ids_2=['task83', 'task84', 'task85', 'task86', 'task87', 'task88'],
                         task_ids_3=['task89', 'task90', 'task91', 'task92', 'task93', 'task94'],
                         output_file='USE2_results/Summarization_metrics.csv'):
    """
    Evaluate text summarization tasks using BLEU and ROUGE metrics
    
    Args:
        file_paths: List of CSV file paths to evaluate
        task_ids: List of primary task IDs
        task_ids_2: List of CT scan task IDs
        task_ids_3: List of MRI task IDs
        output_file: Path to save the results
        
    Returns:
        Dictionary with results for each model and task
    """
    ensure_output_dir()
    result_dict = {}
    
    # Process primary task IDs
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
            if cnt > 0:
                result_dict[model_name][cur_task_id] = (
                    100 * bleu1_sum / cnt, 
                    100 * rouge1_sum / cnt
                )
            else:
                if cur_task_id not in result_dict[model_name].keys():
                    result_dict[model_name][cur_task_id] = (-1, -1)
    
    # Process CT scan task IDs (group 2)
    for filepath in file_paths:
        model_name = filepath.replace('.csv', '').split('_')[-1]
        df = pd.read_csv(filepath)
        
        cnt = 0
        bleu1_sum, rouge1_sum = 0, 0
        
        for cur_task_id in task_ids_2:
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
                
        # Calculate average BLEU and ROUGE scores for CT tasks
        if cnt > 0:
            result_dict[model_name]["CT"] = (
                100 * bleu1_sum / cnt, 
                100 * rouge1_sum / cnt
            )
        else:
            result_dict[model_name]["CT"] = (-1, -1)
    
    # Process MRI task IDs (group 3)
    for filepath in file_paths:
        model_name = filepath.replace('.csv', '').split('_')[-1]
        df = pd.read_csv(filepath)
        
        cnt = 0
        bleu1_sum, rouge1_sum = 0, 0
        
        for cur_task_id in task_ids_3:
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
                
        # Calculate average BLEU and ROUGE scores for MRI tasks
        if cnt > 0:
            result_dict[model_name]["MRI"] = (
                100 * bleu1_sum / cnt, 
                100 * rouge1_sum / cnt
            )
        else:
            result_dict[model_name]["MRI"] = (-1, -1)
    
    # Write results to CSV
    all_task_ids = task_ids + ['CT', 'MRI']
    with open(output_file, 'w') as fp:
        fp.write('Model Name,' + ','.join(all_task_ids) + ', Avg.\n')
        
        for model_name in result_dict.keys():
            fp.write(model_name + ',')
            
            for task_id in all_task_ids:
                fp.write(f"{result_dict[model_name][task_id][0]:.2f}/{result_dict[model_name][task_id][1]:.2f},")
                
            # Calculate and write average BLEU/ROUGE across primary tasks only
            avg_bleu = sum([result_dict[model_name][task_id][0] for task_id in task_ids]) / len(task_ids)
            avg_rouge = sum([result_dict[model_name][task_id][1] for task_id in task_ids]) / len(task_ids)
            fp.write(f"{avg_bleu:.2f}/{avg_rouge:.2f}\n")
    
    return result_dict

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate text summarization tasks")
    parser.add_argument("--files", nargs="+", help="List of CSV files to evaluate")
    parser.add_argument("--primary_task_ids", nargs="+", 
                      default=['task78', 'task114', 'task112', 'task110'], 
                      help="Primary task IDs")
    parser.add_argument("--ct_task_ids", nargs="+", 
                      default=['task83', 'task84', 'task85', 'task86', 'task87', 'task88'], 
                      help="CT scan task IDs")
    parser.add_argument("--mri_task_ids", nargs="+", 
                      default=['task89', 'task90', 'task91', 'task92', 'task93', 'task94'], 
                      help="MRI task IDs")
    parser.add_argument("--output", type=str, default="USE2_results/Summarization_metrics.csv", 
                      help="Output file path")
    
    args = parser.parse_args()
    
    if args.files:
        evaluate_summarization(
            args.files, 
            args.primary_task_ids, 
            args.ct_task_ids, 
            args.mri_task_ids, 
            args.output
        )
