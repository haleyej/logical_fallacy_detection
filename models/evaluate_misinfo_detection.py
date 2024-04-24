import os 
import csv
import torch
import evaluate
import argparse
import numpy as np

from tqdm import tqdm
from typing import Literal
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, BertTokenizerFast, AutoModelForSequenceClassification, DistilBertForSequenceClassification

from liar_fine_tuning import LiarDataset

#to run 
#python3 evaluate_misinfo_detection.py  --data_path='../data/liar_dataset/test.tsv' --model_type='pretrained' --model_checkpoint='../pretrained_models/liar_output_snli/checkpoint-16000'

# set metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(predictions:list[int], labels:list[int]) -> dict:
    '''
    helper function to compute evaluation metrics

    ARGUMENTS: 
        predictions: list of binary class predictions from model 
        labels: list of true binary labels 

    RETURNS:
        dictionary mapping evaluation metrics to their values
    '''
    f1_macro = f1.compute(predictions=predictions, references=labels, average='macro')['f1']
    f1_weighted = f1.compute(predictions=predictions, references=labels, average='weighted')['f1']
    accuracy_score = accuracy.compute(predictions=predictions, references=labels)['accuracy']
    return {'accuracy':accuracy_score, 'f1_macro':f1_macro, 'f1_balanced':f1_weighted}


def evaluate_model(data_path:str, 
                   model_checkpoint:str,
                   save_path:str=os.getcwd(), 
                   model_type:Literal['base', 'pretrained']='base',
                   batch_size:int=8, 
                   max_len:int=512) -> None:
    
    '''
    runs evaluation on misinformation detection test set 
    saves results to file in the `save_path` directory

    ARGUMENTS:
        data_path: path to data 
        save_path: path to save results to
        model_type: base or pretrained 
            - either distilbert model or distilbert + snli fine-tuning 
        model_checkpoint: path to saved model checkpoint
        bath_size: number of points to process at a time
        max_len: maximum length of sequences

    RETURNS:
        None
    '''
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = BertTokenizerFast.from_pretrained('google-bert/bert-base-uncased')

    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
    model = model.to(device)
    model.eval()

    test_data = LiarDataset(data_path, tokenizer, max_len)
    test_loader = DataLoader(test_data, 
                             batch_size=batch_size, 
                             collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding='max_length'))
    
    all_predictions = []
    all_labels = []

    for batch in tqdm(test_loader):
        input_ids = batch['input_ids'].to(device)
        attention_masks = batch['attention_mask'].to(device)
        labels = batch['labels']

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_masks)

        predictions = torch.argmax(outputs.logits, dim=1).tolist()
        all_predictions.extend(predictions)
        all_labels.extend(labels)

    all_predictions = np.array(all_predictions)
    all_labels = torch.stack(all_labels).numpy()
    
    scores = compute_metrics(all_predictions, all_labels)

    with open(os.path.join(save_path, f'{model_type}_evaluation.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['metric', 'value'])
        for metric, value in scores.items():
            writer.writerow([metric, value])

    

def main(args):
    evaluate_model(args['data_path'], 
                   args['model_checkpoint'],
                   args['save_path'],
                   args['model_type'],  
                   args['batch_size'], 
                   args['max_len']) 


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--model_checkpoint', type=str) 
    parser.add_argument('--save_path', type=str, default=os.getcwd())
    parser.add_argument('--model_type', type=str, default='base')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_len', type=int, default=512)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args_dict = vars(args)
    main(args_dict)