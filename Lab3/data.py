import io, sys, math, re
from collections import defaultdict
import numpy as np # type: ignore

class data:
    def load_data(self,filename):
        fin = io.open(filename, 'r', encoding='utf-8')
        data = []
        vocab = defaultdict(lambda:0)
        for line in fin:
            sentence = line.split()
            data.append(sentence)
            for word in sentence:
                vocab[word] += 1
        return data, vocab

    def remove_rare_words(self,data, vocab, mincount = 1):
        data_with_unk = []
       
        for sentence_list in data:
            for i in range(len(sentence_list)):
                if vocab[sentence_list[i]] < mincount:
                    sentence_list[i] = "<unk>"
            data_with_unk.append(sentence_list)
                
        return data_with_unk  