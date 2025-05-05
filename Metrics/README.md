# MedS-Bench Evaluation Framework

This folder contains a comprehensive framework for evaluating the performance on Meds-Bench.

## Overview

The framework provides standardized evaluation metrics for multiple medical task categories:

- Diagnosis
- Treatment Planning
- Natural Language Inference (NLI)
- Named Entity Recognition (NER)
- Medical Explanation
- Medical Reasoning
- Text Summarization
- Text Classification
- Information Extraction
- Discharge Instruction Generation
- Clinical Outcome Prediction
- Fact Verification
- Multiple Choice Question Answering (MCQA)

## Requirements

```
pandas
nltk
rouge
janome
jieba
scikit-learn
```

Install dependencies:

```bash
pip install pandas nltk rouge_score janome jieba scikit-learn
python -c "import nltk; nltk.download('punkt')"
```

## Directory Structure

```
├── utils.py                         # Common utility functions
├── Diagnosis.py                     # Diagnosis task evaluation
├── TreatmentPlanning.py             # Treatment planning task evaluation
├── NatureLanguageInference.py       # NLI classification and generation
├── NamedEntityRecognition.py        # Multi-entity recognition
├── ConceptExplanation.py            # Medical explanation evaluation
├── Rationale.py                     # Medical reasoning evaluation
├── TextSummarization.py             # Medical text summarization
├── TextClassification.py            # Medical text classification
├── InformationExtraction.py         # Information extraction evaluation
├── ClinicalOutcomePrediction.py     # Clinical outcome prediction
├── FactVerification.py              # Medical fact verification
└── MCQA.py                          # Multiple choice question answering
```

## Usage

Each script can be run independently to evaluate specific tasks. All scripts follow a similar command-line interface pattern.

### General Usage Pattern

```bash
python <script_name>.py --files <csv_files> --task_ids <task_ids> --output <output_file>
```

Where:
- `<script_name>`: The name of the task-specific script
- `<csv_files>`: One or more CSV files containing model outputs
- `<task_ids>`: Task IDs to evaluate (optional, defaults provided in each script)
- `<output_file>`: Path to save the evaluation results (optional)

### Input File Format

The input CSV files should have the following columns:
1. `task_id`: The ID of the task
2. `input`: The input prompt/text
3. `GT`: The ground truth answer/reference
4. `output`: The model's output/prediction

### Example Usage

Evaluate diagnosis tasks:

```bash
python diagnosis.py --files ./results/model1.csv ./results/model2.csv --output ./eval_results/diagnosis_results.csv
```

Evaluate NLI tasks (both classification and generation):

```bash
python nli.py --files ./results/model1.csv ./results/model2.csv --type both
```

Evaluate MCQA tasks:

```bash
python mcqa.py --files ./results/model1.csv --task_ids task4 task5 task6
```

## Evaluation Metrics

The framework uses different evaluation metrics depending on the task:

- **Accuracy**: Used for diagnosis, treatment planning, NLI classification, easy NER, information extraction, clinical outcome prediction, and fact verification (classification)
- **F1 Score**: Used for hard NER and text classification
- **BLEU/ROUGE**: Used for explanation, reasoning, summarization, discharge instruction, and fact verification (generation)

## Output Format

Results are saved as CSV files in the specified output directory (defaults to `USE2_results/`). For accuracy-based metrics, the results include the percentage of correct answers. For BLEU/ROUGE-based metrics, the results include both BLEU and ROUGE scores in the format `BLEU/ROUGE`.
