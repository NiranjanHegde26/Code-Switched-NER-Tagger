# README

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
