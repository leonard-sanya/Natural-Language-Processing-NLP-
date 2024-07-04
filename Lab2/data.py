import io, sys
import numpy as np # type: ignore

class data:
    def load_vectors(self, path):
        fin = io.open(path, 'r', encoding='utf-8', newline='\n')
        n, d = map(int, fin.readline().split())
        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = np.asarray([float(x) for x in tokens[1:]])
        return data
    
    def load_lexicon(self, path):
        fin = io.open(path, 'r', encoding='utf-8', newline='\n')
        data = []
        for line in fin:
            a, b = line.rstrip().split(' ')
            data.append((a, b))
        return data