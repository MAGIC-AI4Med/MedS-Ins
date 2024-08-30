# MedS-Bench Leaderboard Submission Guidelines

Welcome to the MedS-Bench Leaderboard Submission Guidelines. In this tutorial, we will guide you through the steps required to submit your model and results for official evaluation on the MedS-Bench platform. Once evaluated, your scores will be showcased on the leaderboard.

## Submission Methods
You can choose from the following two methods to submit your work:

1. **Direct Model Submission**: Share your model or a Hugging Face model link with us for evaluation.
2. **Result Submission**: Evaluate your model using MedS-Bench on your own and submit the resulting CSV file.

### 1. Direct Model Submission
Please include the following in your submission:

- **Model Description**: Provide a brief description of your model.
- **Conda Environment File**: Include a description of the packages used and, preferably, an `environment.yaml` file for easy setup using Anaconda.
- **Inference Script**: Provide a Python script used for model inference.

### 2. Result Submission
Your submission should include:

- **Model Description**: A brief description of your model.
- **Results File**: Submit a CSV file containing the following four headers:
  - `task_id`: A unique identifier for each task, obtained from the filename.
  - `input`: Your input to the LLM.
  - `GT`: The ground truth output from MedS-Bench.
  - `output`: The output from your LLM.
  - Example file: [eval_results_Rationale_1_context_1_oriINS_MMedS-Llama 3.csv](./eval_results_Rationale_1_context_1_oriINS_MMedS-Llama 3.csv), which demonstrates results for the MMedS-Llama 3 on Rationale tasks.

## Removal from the Leaderboard

If you wish to remove your model's scores from the leaderboard, please send an email to our support team. Removals will be processed during the next leaderboard update.
