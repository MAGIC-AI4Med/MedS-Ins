from model import MedS_Llama3
if __name__ == '__main__':
    sdk_api = MedS_Llama3(model_path="../MMedS-Llama-3-8B-v1", gpu_id=1)
    INSTRUCTION = "If you are a doctor, please perform clinical consulting with the patient."
    results = sdk_api.chat([], "What is the treatment for diabetes?", INSTRUCTION)
    print(results)