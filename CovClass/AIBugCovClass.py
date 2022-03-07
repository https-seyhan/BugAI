import  pandas as pd
import numpy as np
import seaborn as sb
from sklearn.covariance import EllipticEnvelope # An object for detecting outliers in a Gaussian distributed dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn import datasets #Import scikit-learn dataset library
from gensim.models import Word2Vec
from numpy import array
from keras.utils import to_categorical
from matplotlib import pyplot as plt

maxlen = 5 #400 # number of words in a row. Input words.
embedding_dims = 6 #300 #5 #300

def convertcbow(dataset):
    sentences = []
    vectorised_codes = []
    ast = [row.split('::') for row in dataset['classname']]
    # the input to the cbow is list of list of each line
    cbowmodel = Word2Vec(ast, min_count=1, size=embedding_dims, workers=3, window=6, sg=0)
    print(' CBOW model ', cbowmodel)
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
   
    # Patch Data
    for _ in range(len(data[0][0])):
        zero_vector.append(0.0)
       
    for sample in data:
        if len(sample) > maxlen:
            temp = sample[:maxlen]
        elif len(sample) < maxlen:
            temp = sample
            additional_elems = maxlen - len(sample)

            for _ in range(additional_elems):
                temp.apembedding_dimspend(temp)
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

def CovModel(vectorised_data, target):
    split_point = int(len(vectorised_data) * .7)
    print('Split Point ', split_point)
    # split data into training and testing
    x_train = vectorised_data[:split_point]
    y_train = target[:split_point]
    #y_train = to_categorical(y_train, 2)
    #plt.hist(x_train)
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
   
    outliers_fraction =6/300
    n_outliers = int(outliers_fraction * nsamples)
    print("Number of Outliners :", n_outliers)

    covmodel = EllipticEnvelope(contamination=0.4)
    covmodel.fit(x_train, y_train)
    pred = covmodel.predict(x_test)
    
    # Model Accuracy: how often is the classifier correct?
    print("Accuracy: {:3f}".format(accuracy_score(y_test, pred > 0.5)))
    # print("Confusion matrix:\n{}".format(confusion_matrix(y_test.argmax(axis=1), pred.argmax(axis=1))))
    print("Confusion matrix:\n{}".format(confusion_matrix(np.array(y_test), pred)))
    print(classification_report(y_test, pred))

if __name__ == '__main__':
    dataset = getDataset()
    vectorised_data = convertcbow(dataset)
    target = collect_expected(dataset)  # Biased two classes {198, 2} lenght is 200
    CovModel(vectorised_data, target)
