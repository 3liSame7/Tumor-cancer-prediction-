import pickle
from LogisticRegression import log
from SVM import svm
from DecisionTree import tree
from RandomForest import forest
import os


def save(model):
    path = input("Enter Path")
    with open(path, 'wb') as file:
        pickle.dump(model, file)
    return path


def load(path):
    # path = save(svm)
    os.path.exists(path)
    model = pickle.load(open(path, 'rb'))
    return model
