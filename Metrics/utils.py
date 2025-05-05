# utils.py
# Common utility functions for all evaluation scripts

import tqdm
import warnings
import pandas as pd
import sys
import json
import jieba
from rouge import Rouge 
from janome.tokenizer import Tokenizer
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import os

# Set recursion limit for complex processing
sys.setrecursionlimit(10000)
# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# BLEU weight configurations
weights_list = [
    (1, 0, 0, 0),
    (0, 1, 0, 0),
    (0, 0, 1, 0),
    (0, 0, 0, 1),
    (0.25, 0.25, 0.25, 0.25),
]

def csv_to_list(file_path):
    """
    Convert CSV file to a list of dictionaries
    
    Args:
        file_path: Path to the CSV file
    
    Returns:
        List of dictionaries with task_id, input, GT and output fields
    """
    df = pd.read_csv(file_path)
    data_list = []
    for index, row in df.iterrows():
        data_dict = {
            'task_id': row[0],
            'input': row[1],
            'GT': row[2],
            'output': row[3]
        }
        data_list.append(data_dict)
    return data_list

def cal_similarity(response_rationale, gt_rationale, lang='en', metric='BLEU'):
    """
    Calculate similarity between two texts using BLEU or ROUGE metrics
    
    Args:
        response_rationale: Model output text
        gt_rationale: Ground truth text
        lang: Language code ('en', 'zh', 'jp', 'spa', 'fra', 'ru')
        metric: Metric type ('BLEU' or 'Rouge')
        
    Returns:
        Similarity score
    """
    if metric == 'BLEU':
        if lang in ['en', 'spa', 'fra', 'ru']:
            reference_tokenized = word_tokenize(gt_rationale)
            candidate_tokenized = word_tokenize(response_rationale)
        elif lang in ['zh']:
            reference_tokenized = list(jieba.cut(gt_rationale))
            candidate_tokenized = list(jieba.cut(response_rationale))
        elif lang in ['jp']:
            t = Tokenizer()
            reference_tokenized = [token.surface for token in t.tokenize(gt_rationale)]
            candidate_tokenized = [token.surface for token in t.tokenize(response_rationale)]

        # Calculate BLEU score
        smoothie = SmoothingFunction().method1
        bleu_score = sentence_bleu([reference_tokenized], candidate_tokenized, smoothing_function=smoothie, weights=(1, 0, 0, 0))
        return bleu_score
    
    elif metric == 'Rouge':
        if lang == 'zh':
            gt_rationale = " ".join(jieba.cut(gt_rationale))
            response_rationale = " ".join(jieba.cut(response_rationale))
        elif lang == 'jp':
            t = Tokenizer()
            gt_rationale = " ".join([token.surface for token in t.tokenize(gt_rationale)])
            response_rationale = " ".join([token.surface for token in t.tokenize(response_rationale)])
        
        rouge = Rouge()
        rouge_scores = rouge.get_scores(response_rationale, gt_rationale)
        
        return rouge_scores[0]['rouge-l']['f']

def ensure_output_dir(directory='USE2_results'):
    """
    Ensure the output directory exists
    
    Args:
        directory: Directory path to create if it doesn't exist
    """
    if not os.path.exists(directory):
        os.makedirs(directory)