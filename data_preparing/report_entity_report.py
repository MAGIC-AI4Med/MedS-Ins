import json

ROOT_DIR = "Report_NER/train_new_new.jsonl"


result_dict = {}
result_dict["Contributors"] = "Weike Zhao"
result_dict["Source"] = "radiopaedio_caption_ner"
result_dict["URL"] = ""
result_dict["Categories"] = ["Sentence Composition"]
result_dict["Definition"] = ["In this task, you have to decompose a medical clinical report sentence. I will input a sentence scratched from a medical report and your goal is to decompose it into Anatomy part, Abnormality part, Non-Abnormality part, or Disease part.In output, you have to organize the answer in a typical python list format, like '[[A,type],[B,type],...]' and please do not destroy the order in which the original part appears in the given sentence. "]
result_dict["Reasoning"] = []
result_dict["Input_language"] = ["English"]
result_dict["Output_language"] = ["English"]
result_dict["Instruction_language"] = ["English"]
result_dict["Domains"] = ["Medicine","Clinical Reports", "Radiology"]
result_dict["Positive Examples"] = []
result_dict["Negative Examples"] = []
result_dict["Instances"] = []

with open(ROOT_DIR,'r') as f:
    for line in f:
        data = json.loads(line)
        input_ss  = data["sentences"][0]
        ner = data['ner'][0]
        output_ss = []
        for kk in ner:
            output_ss.append([' '.join(input_ss[kk[0]:(kk[1]+1)]),kk[2]])
        output_ss = repr(output_ss)
        result_dict["Instances"].append(
                {
                    "input": data["subtext"],
                    "output": output_ss
                }
            )
        
with open("task72_report_entity_extraction.json",'w') as f:
    json.dump(result_dict, f ,indent=4)