import numpy as np # type: ignore

from data import data # type: ignore
from functions import functions # type: ignore

dataloader = data()
word_vectors = dataloader.load_vectors('./wiki.en.vec')
word_vectors_en = dataloader.load_vectors('./wiki.en.vec')
word_vectors_fr = dataloader.load_vectors('./wiki.fr.vec')
lexicon = dataloader.load_lexicon("./lexicon-en-fr.txt")
train = lexicon[:5000]
valid = lexicon[5000:5100]
function = functions(word_vectors)

career = ['executive', 'management', 'professional', 'corporation',
          'salary', 'office', 'business', 'career']
family = ['home', 'parents', 'children', 'family',
          'cousins', 'marriage', 'wedding', 'relatives']
male = ['john', 'paul', 'mike', 'kevin', 'steve', 'greg', 'jeff', 'bill']
female = ['amy', 'joan', 'lisa', 'sarah', 'diana', 'kate', 'ann', 'donna']


def main():
    print('=========================================================================')
    print (f"test similarity {function.cosine(np.array([1,0,0]),np.array([1,0,0]))}")
    print('similarity(apple, apples) = %.3f' %
        function.cosine(word_vectors['apple'], word_vectors['apples']))
    print('similarity(apple, banana) = %.3f' %
        function.cosine(word_vectors['apple'], word_vectors['banana']))
    print('similarity(apple, tiger) = %.3f' %
        function.cosine(word_vectors['apple'], word_vectors['tiger']))
    

    print('=========================================================================')
    print('The nearest neighbor of cat is: ' +
      function.nearest_neighbor(word_vectors['cat'], word_vectors, exclude_words = ['cat', 'cats']))
    
    print('=========================================================================')
    knn_cat = function.knn(word_vectors['cat'], word_vectors, 5)
    print('Best K element')
    print('cat')
    print('=========')
    for score, word in function.knn(word_vectors['cat'], word_vectors, 5):
        print (word + '\t%.3f' % score)

    print('=========================================================================')
    print('Word analogies')       
    print('france - paris + rome = ', function.analogy('paris', 'france', 'rome', word_vectors)) 


    print('Word Biases')
    print('similarity(genius, man) = %.3f' %
        function.cosine(word_vectors['man'], word_vectors['genius']))
    print('similarity(genius, woman) = %.3f' %
        function.cosine(word_vectors['genius'],word_vectors['woman']))
    
    print('=========================================================================')
    print('Word embedding association')
    print('Word embedding association test: %.3f' %
        function.weat(career, family, male, female, word_vectors))   
     
    print('=========================================================================')
    print("Word Translation")
    mapping = function.align(word_vectors_en, word_vectors_fr, lexicon)
    print(function.translate("man", word_vectors_en, word_vectors_fr, mapping))
    print(function.translate("machine", word_vectors_en, word_vectors_fr, mapping))
    print(function.translate("learning", word_vectors_en, word_vectors_fr, mapping))
    
    print('=========================================================================')
    print("Model Evaluation")
    function.evaluate(valid, word_vectors_en, word_vectors_fr, mapping)


if __name__ == "__main__":
  main()