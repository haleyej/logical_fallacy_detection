import os 
import torch
import numpy as np

from tqdm import tqdm
from collections import Counter
from torch.utils.data import Dataset



class LiarDataset(Dataset):
    def init(self, path:str, tokenizer, max_len:int=512) -> None:
        labels_map = {
            'true': 1, 
            'mostly-true': 1, 
            'half-true': 0, 
            'barely-true': 0, 
            'false': 0, 
            'pants-fire': 1
        }

        texts = []
        labels = []
        with open(path) as f:
            lines = f.readlines()
            for line in lines:
                line = line.split("\t")

                texts.append(line[2])
                labels.append(labels_map[line[1]])

        self.tokenizer = tokenizer 
        self.texts = texts
        self.labels = labels
        self.max_len = max_len


    def get_label_distribution(self) -> dict:
        return Counter(self.labels)
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, index:int) -> dict:
        text = self.texts[index]
        label = self.labels[index]

        output_dict = self.tokenizer(
                    text = text, 
                    padding = True, 
                    truncation = True, 
                    max_length = self.max_len, 
                    return_attention_mask = True, 
                    add_special_tokens = True,
                    return_special_tokens_mask = False,
                    return_token_type_ids = False,
                    return_offsets_mapping = False)

        output_dict['label'] = torch.tensor(label)
        return output_dict

    





                

