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

# set device
device = 'cpu' 
if torch.cuda.is_available():
    device = 'cuda'
print(f"Using '{device}' device")

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


# set metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")


def load_logical_fallacies(path:str) -> tuple[list[int]]:
    df = pd.read_csv(path)

    texts = df['source_article'].to_list()
    labels = df['id'].to_list()

    return texts, labels 


def fine_tune_logic(train_data_path:str, 
                    eval_data_path:str,
                    output_dir:str, 
                    logging_dir:str,
                    model_checkpoint:str='distilbert/distilbert-base-uncased',
                    max_len:int=512,
                    epochs:int=20,
                    batch_size:int=8,
                    eval_steps:int=5000,
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
        model_checkpoint: path to pretrained model (locally or on HuggingFace)
        max_len: max length of sequences
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
    tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")

    train_data, train_labels = load_logical_fallacies(train_data_path)
    eval_data, eval_labels = load_logical_fallacies(eval_data_path)

    train_dataset = LogicDataset(tokenizer, train_data, train_labels, max_len)
    eval_dataset = LogicDataset(tokenizer, eval_data, eval_labels, max_len)

    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels = len(set(train_labels)))
    print(model)

    peft_config = LoraConfig(task_type = TaskType.SEQ_CLS, 
                             inference_mode = inference_mode, 
                             r = r, 
                             lora_alpha = lora_alpha, 
                             lora_dropout = lora_dropout, 
                             target_modules = ['lin1', 'lin2', 'pre_classifier', 'classifier'])
    
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
        logging_steps = eval_steps,
        weight_decay = weight_decay,
        warmup_steps = 500,
        save_strategy = "steps",
        save_steps = eval_steps * 2,
        load_best_model_at_end = True
    )

    trainer = Trainer(
        model = model, 
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        tokenizer = tokenizer,
        compute_metrics = compute_metrics
    )

    with tqdm(total=training_args.num_train_epochs, desc="Training") as pbar:
        trainer.train()
        pbar.update(1)

    model_save_path = os.path.join(output_dir, "logic-snli-classification-weights.pth")
    torch.save(model.state_dict(), model_save_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str)
    parser.add_argument('--eval_data_path', type=str)
    parser.add_argument('--output_dir', type=str, default='snli_output')
    parser.add_argument('--logging_dir', type=str, default='snli_logging')
    parser.add_argument('--model_checkpoint', type=str, default='distilbert/distilbert-base-uncased')
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--eval_steps', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--inference_mode', type=bool, default=False)
    return parser.parse_args()


def main(args):
    fine_tune_logic(args['train_data_path'], 
                    args['eval_data_path'], 
                    args['output_dir'], 
                    args['logging_dir'],
                    args['model_checkpoint'], 
                    args['max_len'], 
                    args['epochs'], 
                    args['batch_size'], 
                    args['eval_steps'], 
                    args['lr'], 
                    args['weight_decay'], 
                    args['r'], 
                    args['lora_alpha'], 
                    args['lora_dropout'], 
                    args['inference_mode']) 
    
if __name__ == "__main__":
    args = parse_args()
    args_dict = vars(args)
    main(args_dict)