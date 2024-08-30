import argparse
import csv
import tqdm
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from EvalDataset import InsDataset

# Parse command line arguments
parser = argparse.ArgumentParser(description="Evaluate language models on benchmark tasks.")
parser.add_argument("--model_type", required=True, help="Type of model to evaluate.")
parser.add_argument("--model_path", required=True, help="Path to the pre-trained model.")
parser.add_argument("--benchmark_task", required=True, help="Benchmark task to evaluate on.")
parser.add_argument("--eos_id", default=None, help="End of sequence token id if applicable.")
args = parser.parse_args()

# Default probabilities for dataset configuration
NO_CONTEXT_P = 0.0  # Use 3-shot by default
ORI_INS = 1.0       # Use original instructions by default
EOS_TOKEN_ID = int(args.eos_id) if args.eos_id else None

# Constants and Path Definitions
BENCHMARK_ROOT = '/medai/Datasets/Benchmarks/'
SAVE_ROOT = './prediction_results/'
MODEL_TYPE = args.model_type
MODEL_PATH = args.model_path
BENCHMARK_TASK = args.benchmark_task
SAVE_DIR = f"{SAVE_ROOT}eval_results_{BENCHMARK_TASK}_{int(1 - NO_CONTEXT_P)}_context_{int(ORI_INS)}_oriINS_{MODEL_TYPE}.csv"


# Load model and tokenizer
if MODEL_TYPE != 'Mistral':
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map='auto',
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        model_max_length=2048,
        padding_side="right",
        trust_remote_code=True
    )
else:
    chatbot = pipeline("text-generation", model=MODEL_PATH, max_new_tokens=200, device_map='auto', torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, model_max_length=2048, padding_side="right", trust_remote_code=True)

# Initialize the evaluation dataset
eval_dataset = InsDataset(tokenizer=tokenizer, root_path=BENCHMARK_ROOT + BENCHMARK_TASK, original_ins_possibility=ORI_INS, no_context_possibility=NO_CONTEXT_P, model_type=MODEL_TYPE)

# Writing predictions to CSV
with open(SAVE_DIR, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['task_id', 'input', 'GT', 'output'])

    for sample in tqdm.tqdm(eval_dataset):
        input_sentence = sample['input']
        output_sentence = sample['output']

        if MODEL_TYPE != 'Mistral':
            input_tokens = tokenizer(input_sentence, return_tensors="pt", max_length=tokenizer.model_max_length, truncation=True).to(model.device)
            max_tokens = 1000 if EOS_TOKEN_ID else 200
            greedy_output = model.generate(**input_tokens, max_new_tokens=max_tokens, eos_token_id=EOS_TOKEN_ID)
            prediction = tokenizer.decode(greedy_output[0][input_tokens['input_ids'].shape[1]:], skip_special_tokens=True)
        else:
            greedy_output = chatbot(input_sentence)[0]['generated_text']
            prediction = greedy_output

        writer.writerow([sample['task_id'], input_sentence, output_sentence, prediction])

