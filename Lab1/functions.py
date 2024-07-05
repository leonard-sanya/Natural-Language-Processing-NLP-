import numpy as np # type: ignore
from collections import defaultdict
import io, sys, math, re
from typing import List, Tuple, Dict
from tqdm.notebook import tqdm,trange # type: ignore
import random

class NaiveBayes:
    def count_words(self,data:str)->Dict:
        n_examples = 0
        n_words_per_label = defaultdict(lambda: 0)
        label_counts = defaultdict(lambda: 0)
        word_counts = defaultdict(lambda: defaultdict(lambda: 0.0))
    
        for example in data:
            label, sentence = example
            n_examples += 1
            
            n_words_per_label[label] += len(sentence)
                
            if label not in label_counts.keys():
                label_counts[label] = 1 
            else:
                label_counts[label] += 1
                
            if label not in word_counts.keys():
                for word in sentence:
                    if word not in word_counts.keys():
                        word_counts[label][word] = 1 
                    else:
                        word_counts[label][word] += 1
            else:
                for word in sentence:
                    word_counts[label][word] += 1
                    
        return {'label_counts': label_counts,
                'word_counts': word_counts,
                'n_examples': n_examples,
                'n_words_per_label': n_words_per_label}

    def predict(self,sentence:List, mu:float, label_counts:Dict, word_counts:Dict, n_examples:int, n_words_per_label:Dict)->str:
        best_label = None
        best_score = float('-inf')
    
        for label in word_counts.keys():
            score = 0.0
            prior = label_counts[label] / sum(label_counts.values())
            #P(Class | Word) = P(Class) * P(word | Class)
            
            for word in sentence:
                word_count = word_counts.get(label,0).get(word,0)
                
                # Calculate the likelihood P(word | label) using Laplace smoothing
                likelihood = (word_count + mu) / (n_words_per_label[label] + mu * len(word_counts.keys()))
                
                score += math.log(likelihood)
            score += math.log(prior)
            
            if score > best_score:
                best_score = score
                best_label = label
            
        return best_label

    def compute_accuracy(self,valid_data:str, mu:float, counts:Dict)->float:
        accuracy = 0.0
        for label, sentence in valid_data:
            predicted_label = self.predict(sentence, mu, counts['label_counts'], counts['word_counts'], 
                                      counts['n_examples'], counts['n_words_per_label'])
            if predicted_label == label:
                accuracy += 1
    
        accuracy = accuracy / len(valid_data)
         
        return accuracy 

class LogisticRegression:
    def softmax(self,x:np.ndarray)->np.ndarray:

        c = np.max(x)
        
        log_sum_exp = c + np.log(np.sum(np.exp(x - c),-1,keepdims=True))
    
        return np.exp(x - log_sum_exp)

    def sgd(self,w:np.ndarray, data:List[Tuple], niter:int, lr:float = 0.01)->np.ndarray:
        random.seed(123)
        nlabels, dim = w.shape
        loss_lis = []
    
        for iter in trange(niter):
    
            total_loss = 0.0
            np.random.shuffle(data)
            
            for label,features in data:
                
                label_pred = self.predict(w,features)
                
                loss = -np.log(label_pred[label])
                total_loss += loss
                
                grads = label_pred.copy()
                grads[label] -= 1
                
                w -= lr * np.outer(grads,features)
            avg_loss = total_loss / len(data)
            print(f"Epoch {iter+1}: train loss -----{avg_loss}")
    
        return w
        
    def predict(self,w:np.ndarray, x:np.ndarray)->np.ndarray:
        z = np.dot(w,x)
        label_pred = self.softmax(z)
        
        return label_pred

    def compute_accuracy(self,w:np.ndarray, valid_data:List[Tuple])->float:

        accuracy = 0.0
        for sample in valid_data:
            label,features = sample
            label_pred = self.predict(w,features)
            
            if np.argmax(label_pred) == label:
                accuracy += 1
    
        accuracy = accuracy / len(valid_data)
        
        return accuracy
            
