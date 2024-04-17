import os 
import evaluate
import argparse
import numpy as np 
import pandas as pd 

from peft import get_peft_model, LoraConfig, TaskType
from transformers import Dataset, TFBertTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# set metrics
metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

def load_snli(path:str) -> tuple[list]:
    '''
    helper function to load snli dataset

    ARGUMENTS: 
        path: path to data
    
    RETURNS: 
        a tuple containing 
            1) a list of sentence pairs (premise, hypothesis)
            2) the labels
    '''
    sentence_pairs = []
    labels = []
    labels_to_ids = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
    
    with open(path) as f: 
        lines = f.readlines()
        for line in lines[1:]:
            sentence_pairs.append((line[5], line[6]))
            labels.append(labels_to_ids[line[0]])
    return sentence_pairs, labels


def compute_metrics(pred) -> dict:
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)
    return metrics.compute(predictions=predictions, references=labels)


class SNLIDataset(Dataset):
    def __init__(self):
        pass 

    def __len__(self):
        pass 

    def __getitem__(self):
        pass


def fine_tune_model(train_data_path:str, 
                    eval_data_path:str,
                    output_dir:str, 
                    logging_dir:str,
                    epochs:int=10,
                    batch_size:int=8,
                    eval_steps:int=2000,
                    lr:float=1e-5,
                    weight_decay:float=0.001,
                    r:int=64,
                    lora_alpha:int=32,
                    lora_dropout:float=0.1,
                    inference_mode:bool=False) -> None:
    '''
    loads in pretrained bert model, fine-tunes on SNLI 
    dataset using LoRA

    ARGUMENTS:
        train_data_path: path to training data
        eval_data_path: path to eval data
        output_dir: directory to save outputs to 
        logging_dir: directory to save logging to
        epochs: number of passes through training data
        batch_size: number of instances per batch update
        eval_steps: evaluate model every n steps
        lr: learning rate 
        weight_decay: weight decay
        r: the dimensions of the low-rank LoRA matrix
        lora_alpha: scaling factor of LoRA matrix
        lora_dropout: amount of dropout in LoRA matrix
        inference_mode: if the mode will be used for inference

    RETURNS: 
        None
    '''
    model_checkpoint = "google-bert/bert-base-uncased"

    tokenizer = TFBertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)

    peft_config = LoraConfig(task_type = TaskType.SEQ_CLS, 
                             inference_mode = inference_mode, 
                             r = r, 
                             lora_alpha = lora_alpha, 
                             lora_dropout = lora_dropout)
    
    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir = output_dir, 
        learning_rate = lr, 
        num_train_epochs = epochs,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        evaluation_strategy = "steps",
        eval_steps = eval_steps,
        logging_dir = logging_dir,
        logging_strategy = "steps",
        logging_steps = 50,
        learning_rate = lr,
        weight_decay = weight_decay,
        warmup_steps = 500,
        save_strategy = "steps",
        save_steps = eval_steps,
        load_best_model_at_end = True
    )

    trainer = Trainer(
        model = model, 
        args = training_args,
        #train_dataset = 
        # eval_dataset = 
        tokenizer = tokenizer,
        compute_metrics = compute_metrics
    )


def parse_args():
    parser = argparse.ArgumentParser()

    return parser.parse_args()

def main(args):
    pass 


if __name__ == "__main__":
    args = parse_args()
    args_dict = vars(args)
    main(args_dict)