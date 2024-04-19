import os
import wandb 
import torch
import evaluate
import argparse
import numpy as np 
import pandas as pd 

from tqdm import tqdm
from torch.utils.data import Dataset
from peft import get_peft_model, LoraConfig, TaskType
from transformers import BertTokenizerFast, AutoModelForSequenceClassification, TrainingArguments, Trainer
from snli_fine_tuning import compute_metrics, LogicDataset

# set metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")


def load_logical_fallacies(path:str) -> tuple[list[int]]:
    df = pd.read_csv(path)

    texts = df['source_article'].to_list()
    labels = df['id'].to_list()

    return texts, labels 


def fine_tune_logic(train_data:str):
    pass 


def main(args):
    pass 


def parse_args():
    parser = argparse.ArgumentParser()

    return parser.parse_args 

if __name__ == '__main__':
    args = parse_args()
    args_dict = vars(args)
    main(args_dict)