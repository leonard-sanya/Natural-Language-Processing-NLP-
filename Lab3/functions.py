import numpy as np # type: ignore
from collections import defaultdict
import io, sys, math, re

class functions:
    def build_ngram(self,data, n):
        
        total_number_words = 0
        counts = defaultdict(lambda: defaultdict(lambda: 0.0))
        def _build_ngram(data,n):
            
            counts = defaultdict(lambda: defaultdict(lambda: 0.0))
            for sentence in data:
                sentence = tuple(sentence)
                
                total_number_words = 0
                for i in range(len(sentence) - n):
                    start = i
                    stop = i + n
                    context = tuple(sentence[start:stop])
                    word = sentence[stop]
                    counts[context][word] += 1
                
    
            prob = defaultdict(lambda: defaultdict(lambda: 0.0))
            for context in counts.keys():
            # p(w | context) = count(context, w)/ count(context)
                total_count = sum(counts[ context].values())
                for word in counts[context].keys():
                    score = (counts[context][word])/total_count
                    prob[context][word] = score   
                    
            return prob
        
        
        prob = defaultdict(dict) 
        for i in range(1,n+1):
            prob.update(_build_ngram(data, i))  
        return prob

    def get_prob(self,model, context, w):
        score = None
        context = tuple(context)
        if model.get(context, {}).get(w, 0) != 0:
            score = model[context][w]
        elif len(context) > 0:
            shorter_context = context[:len(context) - 1]
            context = tuple(shorter_context)
            score = self.get_prob(model, shorter_context, w) 
            if score is not None:
                return score
        else:
            score = 1e-10
        return score

    def perplexity(self,model, data, n):
        log_likelihood = 0.0
        num_words = 0
        for sentence in data:
            num_words += len(sentence)
            for i in range(len(sentence)-n):
                start = i 
                stop = i + n
                context = tuple(sentence[start:stop])
                word = sentence[stop]
                
                prob = math.log(self.get_prob(model, context, word))
                log_likelihood += prob
        log_likelihood *= -1 / num_words
        perplexity = math.exp(log_likelihood)
        
        return perplexity
       
    def get_proba_distrib(self,model, context):
        words_and_probs = defaultdict(lambda: 0.0)
        context = tuple(context)
    
        if context in model:
            total_count = sum(model[context].values())
            for word, count in model[context].items():
                probability = count / total_count
                words_and_probs[word] = probability
        else:
            if len(context) > 0:
                shorter_context = context[:len(context) - 1]
                shorter_words_and_probs = self.get_proba_distrib(model, shorter_context)
                words_and_probs.update(shorter_words_and_probs)
        return words_and_probs  

    def generate(self,model):
        sentence = ["<s>"]
        n =10
        while sentence[-1] != "</s>" and len(sentence)< n:
            
            words_and_probs = self.get_proba_distrib(model,sentence)
            x = list(words_and_probs.keys())
            y = list(words_and_probs.values())
            
            word_pred = np.random.choice(x, 1, p = y)
            sentence.append(word_pred[0])
        return sentence
            