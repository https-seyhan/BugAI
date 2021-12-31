import  pandas as pd
import numpy as np
from sklearn import svmfrom sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

#Import scikit-learn dataset library
from sklearn import datasets
from gensim.models import Word2Vec
from numpy import array
from keras.utils import to_categorical

maxlen = 5 #400 # number of words in a row. Input words.
embedding_dims = 6 #300 #5 #300

def convert_to_cbow(dataset):
    sentences = []
    vectorised_codes = []
    ast = [row.split('::') for row in dataset['classname']]
    #The input to the cbow is list of list of each line
    cbowmodel = Word2Vec(ast, min_count=1, size=embedding_dims, workers=3, window=3, sg=0)
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
    bugs = dataset['criticalBugs'] # Training dataset has 8 critical bugs and test dataset has 2. Extremely unbalanced dataset.
  
    for bug in bugs:
        expected.append(bug)
    return expected

def get_Dataset():
    dataset = pd.read_csv('bug-metrics.csv', sep= ',')
    return dataset

def SVMModel(vectorised_data, target):
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
    print("x_train shapes :", nsamples, nx, ny)
    x_train = np.reshape(x_train, (nsamples, nx * ny))
    #x_train = np.reshape(x_train, (len(x_train), maxlen, embedding_dims))
    nsamples, nx, ny = array(x_test).shape
    print("x_test shapes :", nsamples, nx, ny)
    x_test = np.reshape(x_test, (nsamples, nx * ny))
    #x_train = np.reshape(x_train, (len(x_train), maxlen, embedding_dims))
    print("Reshape of X Test :", x_test.shape)
   
    # create SVM model
    #svmmodel = svm.SVC(kernel='poly', degree=8)
    #svmmodel = svm.SVC(kernel='sigmoid')
    one_class_svm_model = svm.OneClassSVM(nu=0.03, kernel="rbf", gamma=0.02)
    one_class_svm_model.fit(x_train, y_train)
    pred = one_class_svm_model.predict(x_test)
    
    #print("Predictions :", pred, '\n')
    #print ("Actual :", np.array(y_test))
   
    #Model Accuracy: how often is the classifier correct?
    #print("Accuracy:", metrics.accuracy_score(y_test, pred))
    print("Accuracy: {:3f}".format(accuracy_score(y_test, pred > 0.5)))
    # print("Confusion matrix:\n{}".format(confusion_matrix(y_test.argmax(axis=1), pred.argmax(axis=1))))
    print("Confusion matrix:\n{}".format(confusion_matrix(np.array(y_test), pred)))
    print(classification_report(y_test, pred))

if __name__ == '__main__':
    dataset = get_Dataset()
    vectorised_data = convert_to_cbow(dataset)
    print(f'Vectorised Data Type {type(vectorised_data)}')
    target = collect_expected(dataset)  # Biased two classes {198, 2} lenght is 200
    SVMModel(vectorised_data,target)
