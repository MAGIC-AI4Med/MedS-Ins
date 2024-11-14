import pandas as pd

data = pd.read_csv('mimic4ed-benchmark-main/master_dataset.csv')

outcome_list =  [_ for _ in list(data.keys()) if 'outcome' in _]

for key in list(data.keys()):
    print(key,data.iloc[0][key])


import json
import tqdm
result_dict = {}
result_dict["Contributors"] = "mimic4ed-benchmark authors"
result_dict["Source"] = "mimic4ed-benchmark"
result_dict["URL"] = "https://github.com/nliulab/mimic4ed-benchmark"
result_dict["Reasoning"] = []
result_dict["Input_language"] = ["English"]
result_dict["Output_language"] = ["English"]
result_dict["Instruction_language"] = ["English"]
result_dict["Positive Examples"] = []
result_dict["Negative Examples"] = []
result_dict["Instances"] = []


result_dict["Categories"] = ["Data to Text"]
result_dict["Definition"] = ["Next I will give an EHR data of a patient, your goal is to predict whether the patient may revisit emergency department in 72 hours. This can hint clinicians to pay more attention on this case."]
result_dict["Domains"] = ["Medicine","EHR","Emergency Department"]
result_dict["Instances"] = []

variable = ["age", "gender", 
            
            "n_ed_30d", "n_ed_90d", "n_ed_365d", "n_hosp_30d", "n_hosp_90d", 
            "n_hosp_365d", "n_icu_30d", "n_icu_90d", "n_icu_365d", 
            
            "triage_pain", "triage_acuity",
            
            "chiefcom_chest_pain", "chiefcom_abdominal_pain", "chiefcom_headache", 
            "chiefcom_shortness_of_breath", "chiefcom_back_pain", "chiefcom_cough", 
            "chiefcom_nausea_vomiting", "chiefcom_fever_chills", "chiefcom_syncope",
            "chiefcom_dizziness",
            
            "cci_MI", "cci_CHF", "cci_PVD", "cci_Stroke", "cci_Dementia", "cci_Pulmonary", 
            "cci_Rheumatic", "cci_PUD", "cci_Liver1", "cci_DM1", "cci_DM2", 
            "cci_Paralysis", "cci_Renal", "cci_Cancer1", "cci_Liver2", "cci_Cancer2", 
            "cci_HIV",
            
            "eci_Arrhythmia", "eci_Valvular", "eci_PHTN",  "eci_HTN1", "eci_HTN2",  
            "eci_NeuroOther", "eci_Hypothyroid", "eci_Lymphoma", "eci_Coagulopathy", 
            "eci_Obesity", "eci_WeightLoss", "eci_FluidsLytes", "eci_BloodLoss", 
            "eci_Anemia", "eci_Alcohol", "eci_Drugs", "eci_Psychoses", "eci_Depression",
            
            "ed_temperature_last", "ed_heartrate_last", "ed_resprate_last", 
            "ed_o2sat_last", "ed_sbp_last", "ed_dbp_last", "ed_los", "n_med", "n_medrecon"]

outcome = "outcome_ed_revisit_3d"

for i in tqdm.tqdm(range(len(data))):
    sample = data.iloc[i]
    input_ss = []
    for key in variable:
        input_ss.append(key + ': '+ str(sample[key]))
    input_ss = '\n'.join(input_ss)
    output_ss = str(sample[outcome])
    result_dict["Instances"].append(
            {
                "input":input_ss,
                "output": output_ss
            }
        )
    #break

with open("task106_mimic4ed_72h_ed_revisit_disposition.json",'w') as f:
    json.dump(result_dict, f ,indent=4)
    