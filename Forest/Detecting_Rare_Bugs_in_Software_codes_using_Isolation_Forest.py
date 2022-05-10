import  pandas as pd
import numpy as np
import seaborn as sb
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn import metrics
from sklearn import datasets
from gensim.models import Word2Vec
from numpy import array
from keras.utils import to_categorical
from matplotlib import pyplot as plt

#The IsolationForest ‘isolates’ observations by randomly selecting a feature and then randomly selecting 
#a split value between the maximum and minimum values of the selected feature.

#NLP parameters
maxlen = 5 #400 # number of words in a row. Input words.
embedding_dims = 6 #300 #5 #300
outliers_fraction =6/300

def convertcbow(dataset):
    sentences = []
    vectorised_codes = []
    ast = [row.split('::') for row in dataset['classname']]
    #The input of the cbow is list of list of each line
    cbowmodel = Word2Vec(ast, min_count=1, size=embedding_dims, workers=3, window=6, sg=0)
    classes = dataset['classname']
    for codes in classes:
        linecode = []
        tokens = codes.split('::')
        # print(tokens)
        sentences.append(tokens)

        for token in tokens:
            try:
                linecode.append(cbowmodel[token])
            except KeyError:
                pass
        vectorised_codes.append(linecode)
    return vectorised_codes

#Append zeros to the end of the sentences if the sentences are short
def pad_trunc(data, maxlen):
    new_data = []
    zero_vector = []
    for _ in range(len(data[0][0])):
        zero_vector.append(0.0)

    for sample in data:
        if len(sample) > maxlen:
            temp = sample[:maxlen]
        elif len(sample) < maxlen:
            temp = sample
            additional_elems = maxlen - len(sample)

            for _ in range(additional_elems):
                temp.append(temp)
        else:
            temp = sample
        new_data.append(temp)
    return new_data

def collect_expected(dataset):
    expected = []
    bugs = dataset['criticalBugs'] # training dataset has 8 critical bugs and test dataset has 2. Extremely unbalanced dataset.
 
    for bug in bugs:
        expected.append(bug)
    return expected

def getDataset():
    dataset = pd.read_csv('bug-metrics.csv', sep= ',')
    return dataset

def ForestModel(vectorised_data, target):
    split_point = int(len(vectorised_data) * .7)
    print('Split Point ', split_point)
    # split data into training and testing
    x_train = vectorised_data[:split_point]
    y_train = target[:split_point]
    x_test = vectorised_data[split_point:]
    y_test = target[split_point:]
   
    #make each point of data of uniform lenght
    x_train = pad_trunc(x_train, maxlen)
    x_test = pad_trunc(x_test, maxlen)

    nsamples, nx, ny = array(x_train).shape
    x_train = np.reshape(x_train, (nsamples, nx * ny))
    nsamples, nx, ny = array(x_test).shape
    print("x_test shapes :", nsamples, nx, ny)
    x_test = np.reshape(x_test, (nsamples, nx * ny))

    n_outliers = int(outliers_fraction * nsamples)
    print("Number of Outliners :", n_outliers)
 
    forestmodel = IsolationForest(contamination="auto", random_state=42)
    forestmodel.fit(x_train, y_train)
    pred = forestmodel.predict(x_test)
    print("Accuracy: {:3f}".format(accuracy_score(y_test, pred > 0.5)))
    print("Confusion matrix:\n{}".format(confusion_matrix(np.array(y_test), pred)))
    print(classification_report(y_test, pred))

if __name__ == '__main__':
    dataset = getDataset()
    vectorised_data = convertcbow(dataset)
    print(f'Vectorised Data Type {type(vectorised_data)}')
    target = collect_expected(dataset)  # Biased two classes {198, 2} lenght is 200
    ForestModel(vectorised_data, target)
