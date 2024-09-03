## Data Uploading Guideline

Everyone can join in our porject. We recomend two ways to send us with your data, i.e., json or excel.

### Json
For each task we need a json file like following:

```
{
  "Contributors": [""],
  "Source": [""],
  "URL": [""],
  "Categories": [""],
  "Definition": [""],
  "Input_language": [""], 
  "Output_language": [""],
  "Instruction_language": [""],  
  "Domains": [""],    
  "Split Type": STR,
  "Instances": [ { "id": "", "input": "", "output": [""]} ],
}
```

```
Contributors: your name;
Source: dataset name;
URL: data sources;
Categories: high level NLP task categories like QA,summarization;
Definition: detail definition for the task;
Input_language: English or Chinese or something else for context input;
Output_language: English or Chinese or something else for answering;
Instruction_language:  English or Chinese or something else for instruction;
Domains: Clarify which medical sub-domain the task involves with, like drug, diagnosis, or content types, like academical papers, daily conversation, exam, and so on;
Split Type: Train or Test or Train/Test. Determine what you want the data to be used for, such as evaluation, training, or both
Instances: main instances;
```
### Excel

Check the file "Task_Contributor.xlsx" to see how to organize your data.

The excel file has two sub-sheets.

In the "Basic Information" sheet, you have to clarify some basic informations like:

```
Contributors: your name;
URL: data sources;

Source: dataset name;
Text Domains: Clarify which medical sub-domain the task involves with, like drug, diagnosis, or content types, like academical papers, daily conversation, exam, and so on;
Categories: high level NLP task categories like QA,summarization;

Input_language: English or Chinese or something else for context input;
Output_language: English or Chinese or something else for answering;
Instruction_language:  English or Chinese or something else for instruction;

Definition: detail definition for the task;
Split Type: Train or Test or Train/Test. Determine what you want the data to be used for, such as evaluation, training, or both
```

In the "Instences" sheet, you can fill in your instances. Each instance contains "Input" and "Output" key elements following the QA format.

### Contact

When you are ready to upload any one of the above files, you can contact us via Github issues or our email and we will process your upload promptly. 

**We sincerely thank everyone who is willing to contribute data！**
