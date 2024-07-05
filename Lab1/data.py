import io, sys, math, re
from collections import defaultdict
from typing import List, Tuple, Dict
import numpy as np # type: ignore
from tqdm.notebook import tqdm,trange # type: ignore
import random

class data:
    def load_data_naive(self,filename:str)->List[Tuple]:
        fin = io.open(filename, 'r', encoding='utf-8')
        data = []
        for line in fin:
            tokens = line.split()
            data.append((tokens[0], tokens[1:]))
        return data

    def load_data_logistic(self,filename:str, word_dict:Dict, label_dict:Dict)->List[Tuple]:
        fin = io.open(filename, 'r', encoding='utf-8')
        data = []
        dim = len(word_dict) #The size of the vocabulary.
        for line in tqdm(fin):
            tokens = line.split() #Consider tokenization by space in this case.
            label = tokens[0]
    
            yi = label_dict[label]
            xi = np.zeros(dim)
            for word in tokens[1:]:
                if word in word_dict:
                    wid = word_dict[word]
                    xi[wid] += 1.0
            data.append((yi, xi))
        return data

    def build_dict(self,filename:str, threshold:int=1)->Tuple[Dict]:
        fin = io.open(filename, 'r', encoding='utf-8')
        word_dict, label_dict = {}, {}
        counts = defaultdict(lambda: 0)
        for line in tqdm(fin):
            tokens = line.split()
            label = tokens[0]
    
            if not label in label_dict:
                label_dict[label] = len(label_dict)
    
            for w in tokens[1:]:
                counts[w] += 1
    
        for k, v in counts.items():
            if v > threshold:
                word_dict[k] = len(word_dict)
        return word_dict, label_dict