import os 
import torch
import evaluate
import argparse
import numpy as np

from tqdm import tqdm
from typing import Literal
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, BertTokenizerFast, AutoModelForSequenceClassification, DistilBertForSequenceClassification

from liar_fine_tuning import LiarDataset
from snli_fine_tuning import compute_metrics

# set device
device = 'cpu' 
if torch.cuda.is_available():
    device = 'cuda'
print(f"Using '{device}' device")

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

#to run 
#python3 evaluate_misinfo_detection.py  --data_path='../data/liar_dataset/test.tsv' --model_type='pretrained' --model_checkpoint='../pretrained_models/liar_output_snli/checkpoint-16000'
def evaluate_model(data_path:str, 
                   model_type:Literal['base', 'pretrained'],
                   model_checkpoint:str=None,
                   batch_size:int=8, 
                   max_len:int=512) -> None:
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = BertTokenizerFast.from_pretrained('google-bert/bert-base-uncased')

    if model_type == 'pretrained':
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
    elif model_type =='base':
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

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
    
    scores = compute_metrics(all_predictions)
    print(scores)
    

def main(args):
    print(args)
    evaluate_model(args['data_path'], 
                   args['model_type'], 
                   args['model_checkpoint'], 
                   args['batch_size'], 
                   args['max_len']) 


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--model_checkpoint', type=str, default=None) 
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_len', type=int, default=512)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args_dict = vars(args)
    main(args_dict)