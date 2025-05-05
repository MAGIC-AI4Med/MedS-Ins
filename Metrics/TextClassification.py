# text_classification.py
# Evaluation script for text classification tasks

import argparse
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from utils import ensure_output_dir

def evaluate_text_classification(file_paths, task_ids=['task106'],
                               classifications=[
                                   "sustaining proliferative signaling",
                                   "evading growth suppressors",
                                   "resisting cell death",
                                   "enabling replicative immortality",
                                   "inducing angiogenesis",
                                   "activating invasion and metastasis",
                                   "genomic instability and mutation",
                                   "tumor promoting inflammation",
                                   "cellular energetics",
                                   "avoiding immune destruction"
                               ],
                               output_file='USE2_results/Text_classification_metrics.csv'):
    """
    Evaluate text classification tasks using precision, recall, and F1 score
    
    Args:
        file_paths: List of CSV file paths to evaluate
        task_ids: List of task IDs to evaluate
        classifications: List of class names to look for
        output_file: Path to save the results
        
    Returns:
        Dictionary with results for each model
    """
    ensure_output_dir()
    result_dict = {}
    
    def get_classes(text):
        """Helper function to extract class names from text"""
        res_classes = []
        for cls_name in classifications:
            if cls_name in text:
                res_classes.append(cls_name)
        return res_classes
    
    for filepath in file_paths:
        model_name = filepath.replace('.csv', '').split('_')[-1]
        result_dict[model_name] = {}
        
        df = pd.read_csv(filepath)
        
        for cur_task_id in task_ids:
            y_true = []
            y_pred = []
            
            for index in range(len(df)):
                sample = df.iloc[index]
                task_id = sample['task_id']
                
                if task_id != cur_task_id:
                    continue
                    
                response = str(sample['output']).lower().strip()
                gt_answer = str(sample['GT']).lower().strip()
                
                # Extract classes from text
                response_classes = get_classes(response)
                gt_classes = get_classes(gt_answer)
                
                y_true.append(gt_classes)
                y_pred.append(response_classes)
                
            # Skip if no samples found for this task
            if not y_true:
                continue
                
            # Binarize the multi-label data
            mlb = MultiLabelBinarizer(classes=classifications)
            y_true_binary = mlb.fit_transform(y_true)
            y_pred_binary = mlb.transform(y_pred)

            # Calculate metrics
            precision = precision_score(y_true_binary, y_pred_binary, average='macro', zero_division=0)
            recall = recall_score(y_true_binary, y_pred_binary, average='macro', zero_division=0)
            f1 = f1_score(y_true_binary, y_pred_binary, average='macro', zero_division=0)
            
            # Store metrics as percentages
            result_dict[model_name] = {
                'precision': 100 * precision,
                'recall': 100 * recall,
                'f1_score': 100 * f1
            }
    
    # Write results to CSV
    with open(output_file, 'w') as fp:
        fp.write('Model Name,precision,recall,f1_score\n')
        
        for model_name in result_dict.keys():
            fp.write(model_name + ',')
            
            for metric_name in ['precision', 'recall', 'f1_score']:
                fp.write(f"{result_dict[model_name][metric_name]:.2f},")
                
            fp.write("\n")
    
    return result_dict

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate text classification tasks")
    parser.add_argument("--files", nargs="+", help="List of CSV files to evaluate")
    parser.add_argument("--task_ids", nargs="+", default=['task106'], help="Task IDs to evaluate")
    parser.add_argument("--output", type=str, default="USE2_results/Text_classification_metrics.csv", 
                      help="Output file path")
    
    args = parser.parse_args()
    
    if args.files:
        evaluate_text_classification(args.files, args.task_ids, output_file=args.output)
