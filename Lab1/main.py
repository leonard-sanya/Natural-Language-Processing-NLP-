import numpy as np # type: ignore
from collections import defaultdict
from data import data # type: ignore
from functions import NaiveBayes,LogisticRegression # type: ignore

dataloader = data()
model_logistic = LogisticRegression()
model = NaiveBayes()
 
mu = 0.001

def main():
    model_type = input(" Select 1 for Naive Bayes and 2 for Logistic regression: ")
    if model_type == "1":
        dataset_type = input(" Select 1 for dataset1 and 2 for for dataset2: ")
        if dataset_type == "1":
            print('====================== Naive Bayes =======================')
            train_data = dataloader.load_data_naive("train1.txt")
            valid_data = dataloader.load_data_naive("valid1.txt")
        else:
            print('====================== Naive Bayes =======================')
            train_data = dataloader.load_data_naive("train2.txt")
            valid_data = dataloader.load_data_naive("valid2.txt")
    
        counts = model.count_words(train_data)
        print("Validation accuracy: %.3f" % model.compute_accuracy(valid_data, mu, counts))

    elif model_type == "2":
        dataset_type = input(" Select 1 for dataset1 and 2 for for dataset2: ")
        if dataset_type == "1":
            print('====================== Logistic regression =======================')
            word_dict, label_dict = dataloader.build_dict("train1.txt")
            train_data = dataloader.load_data_logistic("train1.txt", word_dict, label_dict)
            valid_data = dataloader.load_data_logistic("valid1.txt", word_dict, label_dict)

        else:
            print('====================== Logistic regression =======================')
            word_dict, label_dict = dataloader.build_dict("train2.txt")
            train_data = dataloader.load_data_logistic("train2.txt", word_dict, label_dict)
            valid_data = dataloader.load_data_logistic("valid2.txt", word_dict, label_dict)

        nlabels = len(label_dict)

        dim = len(word_dict)
        w = np.zeros([nlabels, dim])
        w = model_logistic.sgd(w, train_data, 25)
        print("Validation accuracy: %.3f" % model_logistic.compute_accuracy(w, valid_data))

    else:
        print("Error!, your selection is out of range")

if __name__ == "__main__":
  main()