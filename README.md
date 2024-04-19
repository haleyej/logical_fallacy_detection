# Logical Fallacy Detection

Haley Johnson (haleyej@umich.edu). University of Michigan. 

## Objective
Misinformation has eroded trust in public institutions, elections, and digital platforms. Although significant work has been done on misinformation detection, it is still an open problem in Natural Language Processing. This project leverages natural language inference to fine-tune a BeRT model to classify logical relationships and logical fallacies. Then, I apply this model to a fake news dataset to asses if natural language inference abilities can help LLMs better detect misinformation. 

## Data
This project utilizes data from 3 sources. Due to their large size they are not hosted in this repository, but are publically avaliable:
* [Stanford Natural Language Inference (SNLI) Corpus](https://nlp.stanford.edu/projects/snli/)
* [LOGIC Logical Fallacy Dataset](https://arxiv.org/abs/2202.13758)
* [LIAR Fake News Dataset](https://aclanthology.org/P17-2067/)


## Repository Structure 
```
├── models                  <- Code for fine tuning models
├── papers                  <- Academics papers used for this project
├── project_proposal        <- Project proposal document
├── project_update          <- Project update document &  code
├── LICENSE
└── README.md
```