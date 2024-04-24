# Logical Fallacy Detection

Haley Johnson (haleyej@umich.edu). Course project for EECS 595, Natural Language Processing, at the University of Michigan

## Objective
Misinformation has eroded trust in public institutions, elections, and digital platforms. Although significant work has been done on misinformation detection, it is still an open problem in Natural Language Processing. This project leverages natural language inference to fine-tune a BeRT model to classify logical relationships and logical fallacies. Then, I apply this model to a fake news dataset to asses if natural language inference abilities can help LLMs better detect misinformation. 

## Data
This project utilizes data from 3 sources. Due to their large size they are not hosted in this repository, but are publically avaliable:
* [Stanford Natural Language Inference (SNLI) Corpus](https://nlp.stanford.edu/projects/snli/)
* [LOGIC Logical Fallacy Dataset](https://arxiv.org/abs/2202.13758)
* [LIAR Fake News Dataset](https://aclanthology.org/P17-2067/)


## Repository Structure 
```
├── evaluation              <- Code to visualize and evaluate results
|   └── misinfo_test        <- Evaluation metrics on misinformation test set 
│   └── runs                <- CSV exports from Weights and Biases  
│   └── visualize_runs.py 
│
├── models                  <- Code for runing & fine tuning models
│   └── baselines.ipynb     
│   └── evaluate_misinfo_detection.py 
│   └── liar_fine_tuning.py 
│   └── logic_fine_tuning.py 
│   └── run_snli.sh
│   └── snli_fine_tuning.py 
│
├── papers                  <- Academics papers used for this project
├── project_proposal        <- Project proposal document
├── project_update          <- Project update document & code
├── LICENSE
└── README.md
```