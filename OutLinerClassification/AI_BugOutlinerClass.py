import  pandas as pd
import numpy as np
import seaborn as sb
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_ dataset['classname']
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report # model quality check
from sklearn import metrics
#Import scikit-learn dataset library
from sklearn import datasets
from gensim.models import Word2Vec
from numpy import array
from keras.utils import to_categorical
from matplotlib import pyplot as plt
maxlen = 5 #400 # number of words in a row. Input words.
embedding_dims = 6 #300 #5 #300

def convertcbow(dataset):
    sentences = []
    vectorised_codes = []
    print("Cbow called")
    ast = [row.split('::') for row in dataset['classname']]
    # Input for the cbow is list of each line
    # size of the word vector of a given token must be equal to embedding_dim of the LocalOutlierFactor model
    cbowmodel = Word2Vec(ast, min_count=1, size=embedding_dims, workers=3, window=6, sg=0)

    # Test cbow model 
    classes = dataset['classname']

    for codes in classes:
        linecode = []
        tokens = codes.split('::')
        sentences.append(tokens)
       
        for token in tokens:
            try:
                linecode.append(cbowmodel[token])
            except KeyError:
                pass
 
        vectorised_codes.append(linecode)
    return vectorised_codes

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
    #bugsdata = pd.read_csv('bug-metrics.csv', sep= ',')
    bugs = dataset['criticalBugs'] # training dataset has 8 critical bugs and test dataset has 2. Extreamly unbalanced dataset.
   
    for bug in bugs:
        expected.append(bug)
    return expected

def getDataset():
    dataset = pd.read_csv('bug-metrics.csv', sep= ',')
    return dataset

def OutlinerModel(vectorised_data, target):
    split_point = int(len(vectorised_data) * .7)
    print('Split Point ', split_point)
    # split data into training and testing
    x_train = vectorised_data[:split_point]
    y_train = target[:split_point]
    #y_train = to_categorical(y_train, 2)
    x_test = vectorised_data[split_point:]
    y_test = target[split_point:]
 
    #make each point of data of uniform lenght
    x_train = pad_trunc(x_train, maxlen)
    x_test = pad_trunc(x_test, maxlen)
    nsamples, nx, ny = array(x_train).shape
    #print("x_train shapes :", nsamples, nx, ny)
    x_train = np.reshape(x_train, (nsamples, nx * ny))
    #x_train = np.reshape(x_train, (len(x_train), maxlen, embedding_dims))
  
    nsamples, nx, ny = array(x_test).shape
    print("x_test shapes :", nsamples, nx, ny)
    x_test = np.reshape(x_test, (nsamples, nx * ny))
    #x_train = np.reshape(x_train, (len(x_train), maxlen, embedding_dims))
    print("Reshape of X Test :", x_test.shape)

    outliers_fraction =6/300
    n_outliers = int(outliers_fraction * nsamples)
    #print("Number of Outliners :", n_outliers)
  
    outlinermodel = LocalOutlierFactor(
        n_neighbors=2, contamination=0.01)

    outlinermodel.fit(x_train, x_test)
    pred = outlinermodel.fit_predict(x_test)
    #y_test = to_categorical(y_test, 2)
    #Model Accuracy: how often is the classifier correct?
    #print("Accuracy:", metrics.accuracy_score(y_test, pred))
    print("Accuracy: {:3f}".format(accuracy_score(y_test, pred > 0.5)))
    # print("Confusion matrix:\n{}".format(confusion_matrix(y_test.argmax(axis=1), pred.argmax(axis=1))))
    print("Confusion matrix:\n{}".format(confusion_matrix(np.array(y_test), pred)))
    print(classification_report(y_test, pred))

if __name__ == '__main__':
    dataset = getDataset()
    vectorised_data = convertcbow(dataset)
    print(f'Vectorised Data Type {type(vectorised_data)}')
    target = collect_expected(dataset)  # Biased two classes {198, 2} lenght is 200
    OutlinerModel(vectorised_data, target)
