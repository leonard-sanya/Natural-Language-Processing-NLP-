import io, sys
import numpy as np # type: ignore
import heapq

class functions:
    def __init__(self,wordvector):
        self.wordvector = wordvector
    
    def cosine(self,u, v):      
        return np.divide(np.dot(u,v) ,np.linalg.norm(u) * np.linalg.norm(v))

    def nearest_neighbor(self,x, word_vectors, exclude_words=[]):
        best_score = -1.0
        best_word = None
    
        for word in word_vectors:
            if word not in exclude_words:
                simalirity_score = self.cosine(x , word_vectors[word]) 
            if simalirity_score > best_score:
                best_score = simalirity_score
                best_word = word
                
        return best_word

    def knn(self,x, vectors, k):
    
        k_nearest_neighbors = []
    
        for word in vectors:
            if not np.array_equal(vectors[word], x):
                simalirity_score = self.cosine(x , vectors[word])
            
                if len(k_nearest_neighbors)!= k:
                    heapq.heappush(k_nearest_neighbors, (simalirity_score,word))
                else:
                    if k_nearest_neighbors[0][0] < simalirity_score:
                        _ = heapq.heappop(k_nearest_neighbors)
                        heapq.heappush(k_nearest_neighbors, (simalirity_score, word))
                    else:
                        pass
            
        return sorted(k_nearest_neighbors, key=lambda x: x[0], reverse=True)     
      

    def analogy(self,a, b, c, word_vectors):
    
        d = (word_vectors[b]/np.linalg.norm(word_vectors[b]) - word_vectors[a]/np.linalg.norm(word_vectors[a])) + word_vectors[c]/np.linalg.norm(word_vectors[c])
    
        desired_key = self.nearest_neighbor(d, word_vectors, exclude_words = [a,b,c])
        return desired_key


    def association_strength(self,w, A, B, vectors):
        strength = 0.0
        part_a = 0.0
        part_b = 0.0
       
        word_vec = vectors.get(w)
        
        for word_a in A:
            word_a_vec = vectors.get(word_a)
            if word_a_vec is not None:
                part_a += self.cosine(word_a_vec,word_vec)
    
        for word_b in B:
            word_b_vec = vectors.get(word_b)
            if word_b_vec is not None:
                part_b += self.cosine(word_b_vec,word_vec)
         
        part_a /= len(A)
        part_b /= len(B)   
        strength = part_a - part_b
    
        return strength


    def weat(self,X, Y, A, B, vectors):
    
        score = 0.0
        part_x = 0.0
        part_y = 0.0
    
        for word_x in X:
            part_x += self.association_strength(word_x,A,B,vectors)
        for word_y in Y:
            part_y += self.association_strength(word_y,A,B,vectors)
            
        score = part_x - part_y
        
        return score

    
    def align(self,word_vectors_en, word_vectors_fr, lexicon):

        x_en, x_fr = [], []
    
        for word_en,word_fr in lexicon:
            if word_en in word_vectors_en and word_fr in word_vectors_fr:
                    x_en.append(word_vectors_en[word_en])
                    x_fr.append(word_vectors_fr[word_fr]) 
                    
        X_en = np.stack(x_en)
        X_fr = np.stack(x_fr)
        
        mapping,_,_,_ = np.linalg.lstsq(X_en,X_fr,rcond = None)
        return mapping

    def translate(self,word, word_vectors_en, word_vectors_fr, mapping):
        desired_value = word_vectors_en[word] @ mapping
        
        desired_key = self.nearest_neighbor(desired_value, word_vectors_fr, exclude_words = [])
           
        return desired_key

    def evaluate(self,valid, word_vectors_en, word_vectors_fr, mapping):

        acc, n = 0.0, 0
        for word_en ,word_fr in valid:
            pred_word_fr = self.translate(word_en,word_vectors_en,word_vectors_fr,mapping)
            if word_fr == pred_word_fr:
                acc +=1
            else:
                pass
            n +=1
        print("Accuracy:", acc / n)
        return acc / n

