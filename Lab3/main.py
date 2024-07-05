import numpy as np # type: ignore
from collections import defaultdict
from data import data # type: ignore
from functions import functions # type: ignore

dataloader = data()
models = functions()

print("load training set..")
train_data, vocab = dataloader.load_data("./train1.txt")
valid_data, _ = dataloader.load_data("./valid1.txt")

train_data2, vocab = dataloader.load_data("./train2.txt")
valid_data2, _ = dataloader.load_data("./valid2.txt")

print("remove rare words")
train_data = dataloader.remove_rare_words(train_data, vocab, mincount = 2)
valid_data = dataloader.remove_rare_words(valid_data, vocab, mincount = 1)


def main():
    print('=========================================================================')
    n = 10
    print("build ngram model with n = ", n)
    model_1 = models.build_ngram(train_data, n)

    print('=========================================================================')
    print("The perplexity is", models.perplexity(model_1, valid_data, n=n))

    print('=========================================================================')
    print("Generated sentence: ",models.generate(model_1))


    n = 3
    model_2 = models.build_ngram(train_data2, n)
    print('=========================================================================')
    print("The perplexity is", models.perplexity(model_2,valid_data2, n))
    print('=========================================================================')
    print("Generated sentence: ",models.generate(model_2))


if __name__ == "__main__":
  main()