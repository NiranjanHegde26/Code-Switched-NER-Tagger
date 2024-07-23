## Introduction
This repository is part of the Computational Linguistics course offered by Saarland University for the Winter Semester of 2023/24. Globalization has encouraged the people to be bilingual/multilingual and the influence of having fluency in multiple languages is visible all across including various social media. People often tend to use two or more languages either in the same sentence or in the discourse. This phenomenon is called Code Switching. The aim of the project, thus is to use neural models to identify NER in the code switched data in English-Spanish context. I have employed XLM-RoBERTa (https://huggingface.co/FacebookAI/xlm-roberta-base) as a neural architecture, added extra head for NER tagging and finetuned the network using 2018 CALCS Dataset (https://ritual.uh.edu/lince/datasets). Further information on the implementation, observation and result is available in the report. So happy to see you stumbling on my small work and hope you like it and have some constructive feedback!

## Directory Structure
```
project-root/
│
├── config.py
├── config.yaml
├── main.ipynb
├── requirements.txt
├── structure.txt
│
├── images/
│   ├── AugDataTrainingLoss.png
│   ├── AugDataValidAcc.png
│   ├── AugDataValidF1.png
│   ├── AugmentedDataTrainingAccuracy.png
│   ├── AugmentedDataTrainingF1.png
│   ├── OriginalDataLoss.png
│   ├── OriginalDataTrainAcc.png
│   ├── OriginalDataTrainF1.png
│   ├── OriginalDataValidationAcc.png
│   ├── OriginalDataValidationF1.png
│   ├── structure.txt
│   ├── Two-shot prompt.png
│   └── Zero-shot prompt.png
│
├── models/
│   └── NERModel.py
│
└── utils/
    ├── dataset_exporter.py
    ├── data_generator.py
    ├── helpers.py
    └── tokenizer.py

```



## Instructions to use
1. Prerequiste: Python 3.10
2. Download the contents into a directory and checkout to the root.
3. Create a new virtual environment using venv with below commands
   Windows:  
   `python -m venv venv_name`

   MacOs/Linux:  
   `python3 -m venv venv_name`
4. Activate the virtual environment  using below commands
   Windows:  
   `venv_name\Scripts\activate`

   MacOs/Linux:  
   `source /venv_name/bin/activate`
5. Install all the dependencies using the below command:  
    `pip install -r requirements.txt`
6. Execute the cells from main.ipynb in the sequential manner.
#### NOTE:
If Google Colab is being used, upload the contents of zip folder to local drive root. After connecting to CPU/GPU execute the first cell to create virtual environment and install all the dependencies.

## Versions
- datasets: 2.16.1
- nltk: 3.8.1
- python: 3.10
- PyYAML: 6.0.1 
- scikit-learn: 1.2.2
- sentencepiece: 0.1.99
- torch: 2.2.1+cu121
- transformers: 4.38.2
- wandb: 0.16.4

## Runtime, accuracy and implementation
All these are listed, explained in the source code as comment lines and text cells.
