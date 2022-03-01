import numpy as np
import datetime
import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
from random import shuffle, sample
from keras.models import Sequential
from keras.utils import to_categorical
from keras import optimizers
#create sample weights
from sklearn.utils import compute_sample_weight
from sklearn.utils import compute_class_weight
from keras.layers import Dense, Activation, Dropout, Flatten, Embedding, LSTM, Bidirectional
from keras.optimizers import RMSprop, Adam, SGD
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from gensim.models import Word2Vec

#The objective iof this model is to predict if there is critical bug in a given AST code using LSTM
#This model does not use pre-trainined embeddings since there is no tra-preined weight vectors are available.
#Therefore, embedding modules is excluded in the LSTM architecture.

maxlen = 5 #400 # number of words in a row. Input words.
batch_size = 200 #32 #32 #200 gave the best results
embedding_dims = 500 #300 #5 #300
epochs = 150 #best result is with 20
num_neurons = 251 #50
sample_size =  997 #10

def getcbow(dataset):
    sentences = []
    vectorised_codes = []
    print("Cbow called")
    #bugs = pd.read_csv('bug-metrics.csv', sep= ',')
    #print(bugs.columns)
    ast = [row.split('::') for row in dataset['classname']]
  
    #The input to the cbow is list of list of each line
    #Size of the word vector of a given token must be equal to embedding_dim of the LSTM model
    cbowmodel = Word2Vec(ast, min_count=1, size= embedding_dims, workers=3, window=3, sg=0)
    print (' CBOW model ', cbowmodel)
    print(cbowmodel['eclipse'])
    classes = dataset['classname']
   
    for codes in classes:
        linecode = []
        tokens = codes.split('::')
        #print(tokens)
        sentences.append(tokens)
       
        for token in tokens:
            try:
                #linecode.append(token)
                #print("Word Vector ", len(cbowmodel[token]))
                linecode.append(cbowmodel[token])
            except KeyError:
                pass
        vectorised_codes. append(linecode)
    #print('Line codes ', linecode)
    #print('Vectorised Codes ', vectorised_codes[0])
    #print('Vectorised Codes ', len(vectorised_codes))
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

def lstmModel(vectorised_data, target):
    split_point =  int(len(vectorised_data) * .7)
    print('Split Point ', split_point)
    #split data into training and testing
    x_train = vectorised_data[:split_point]
    y_train = target[:split_point]
   
    x_test = vectorised_data[split_point:]
    y_test = target[split_point:]
    #make each point of data of uniform lenght
    x_train = pad_trunc(x_train, maxlen)
    x_test = pad_trunc(x_test, maxlen)
    #reshape data into a numpy structure
    print("X_TRAIN Reshape Started ")
    print(f' Training data Size: {len(x_train)}')
    print("Number of word tokens ", maxlen)
    print("Embedding Dims ", embedding_dims)
    #print(f'Training Data {x_train[:1]}')
    
    #y_train = np.array(y_train)
    x_train = np.reshape(x_train, (len(x_train), maxlen, embedding_dims))
    print("X_TRAIN Reshape Completed ")
    #y_train = np.array(y_train)
    y_train = to_categorical(y_train, 2)

    x_test = np.reshape(x_test, (len(x_test), maxlen, embedding_dims))
    #y_test = np.array(y_test)
    y_test = to_categorical(y_test, 2)
    #print(f'Y_TEST DATA: {y_test}')
    print((f'Y_TEST_DATA LENGHT{len(y_test)}'))
    print("Data Reshape Ended ")

    model = Sequential()
    #model.add(Embedding(embedding_dims, batch_size))
    #model.add(Embedding(output_dim=512, input_dim=10000, input_length=100))
    #print(f' Training data Size: {len(x_train)}')

    #model.add(LSTM(num_neurons, return_sequences=True, input_shape=(maxlen, embedding_dims)))
    #model.add(LSTM(num_neurons, return_sequences=True, input_shape=(maxlen , embedding_dims)))
    #model.add(Dense(2, activation='relu'))
    #model.add(Dense(2, activation='sigmoid'))  # one class
    model.add(Bidirectional(LSTM(num_neurons, return_sequences=True, input_shape=(maxlen , embedding_dims)))) #stack LSTMs
    model.add(Dropout(.2))
    model.add(Flatten()) # dense layer expects a flat vectors of n elements
    #model.add(Dense(1, activation='relu')) # one class
    #model.add(Dense(2, activation='elu'))  # one class model.add(Dense(1, activation='relu')) # one class
    #model.add(Dropout(0.2))

    model.add(Dense(2, activation='sigmoid'))  # two class
   
    #model.add(Dense(2, activation='tanh'))  # two class
    #model.add((Dense(2)))
    #model.add(Activation('softmax'))  # one class

    rmsprob = RMSprop(learning_rate=0.0001, rho=0.4) # use learning rate to improve the accuracy of the model
    adam = Adam(lr=0.001)
    #sgd = SGD(lr=0.1)
    #sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.2, nesterov=True)
    
    model.compile(loss='binary_crossentropy', optimizer= rmsprob, metrics=['accuracy'])
    #model.compile(loss='binary_crossentropy', optimizer=adam)
    #model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'] )

    #model.compile('rmsprop', 'binary_crossentropy', metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer=rmsprob)
    #model.compile('rmsprop', 'binary_crossentropy', metrics=['accuracy'])
    #model.compile(sample_weight_mode="temporal"
        #optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #print(model.summary())
    fitmodel(model, x_train, y_train, x_test, y_test, batch_size, epochs)

def fitmodel(model, x_train, y_train, x_test, y_test, batch_size, epochs):
    # print('X Train ', x_train.shape)
    # print(' Y Train ', y_train.shape)
    # print('X Test ', x_test.shape)
    print('Y Test ', y_test.shape)
    #print(y_test)
    #print(np.unique(y_train))
    cls_weight_dict = [{0: 1, 1: 1}, {0: 1, 1: 80}] #two class mapping of weights
    val_sample_weights = compute_sample_weight(cls_weight_dict, y_test)

    weights = compute_sample_weight(class_weight="balanced", y=y_train)
    #weights = compute_sample_weight(class_weight="None", y=y_train)
    #class_weights = compute_class_weight('balanced', y_train,  y_train)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              #sample_weight = weights,
              class_weight = {0 : 1. , 1: 80.},
              #class_weight={0: 1., 1: 100.},
              validation_data=(x_test, y_test, val_sample_weights))

    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              #sample_weight=weights,
              class_weight={0: 1., 1: 80.},
              #class_weight={0: 1., 1: 100.},
              validation_data=(x_test, y_test, val_sample_weights))
    # model.fit(np.array(x_train), np.array(y_train),binary
    # batch_size = batch_size,
    # epochs =epochs,
    # validation_data=(np.array(x_test), np.array(y_test)))

    model_structure = model.to_json()
    with open("BugAITwoClass_model.json", "w") as json_file:
        json_file.write(model_structure)

    conf_matrix(history, model, x_test, y_test)
    #plotresults(history, y_train)
    # model.save_weights("rnn_weights.h5)

    # print(fittedmodel)



def collect_expected(dataset):
    expected = []
    #bugsdata = pd.read_csv('bug-metrics.csv', sep= ',')
    #print(dataset.columns)

    bugs = dataset['criticalBugs'] # training dataset has 8 critical bugs and test dataset has 2. Extreamly unbalanced dataset.

    for bug in bugs:
        #print(bug)
        expected.append(bug)

    return expected

def getDataset():

    dataset = pd.read_csv('bug-metrics.csv', sep= ',')

    #keep = ['classname', 'bugs']
    dataset = dataset.sample(n= sample_size, replace= True, random_state=1)

    #shuffle(dataset)
    #dataset.to_csv('sampledataset.csv')
    return dataset

def getmaxlen(dataset):
    ast = [row.split('::') for row in dataset['classname']]
    print('AST lenght ',len(ast))
    #print(ast[:2])
    
    classes = dataset['classname']

    for codes in classes:

        linecode = []
        tokens = codes.split('::')
        #print(len(tokens))
        

def plotresults(history, y_train):

    weights = compute_sample_weight(class_weight="balanced", y=y_train)

    #print("Weights :::", weights)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']


    epoch = range(1, len(acc) + 1)

    plt.plot(epoch, acc, 'bo', label = 'Training acc')
    plt.plot(epoch, val_acc, 'b', label='Validation acc')

    plt.title('Training and Validation accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epoch, loss, 'bo', label = 'Training loss')
    plt.plot(epoch, val_loss, 'b', label = 'Validation loss')
    plt.title('Training and Validation loss')
    plt.legend()
    plt.show()

def conf_matrix(history, model, x_test, y_test):

    pred = model.predict(x_test)
    #print(f' Predictions on Validation Data: {pred}')
    print("Accuracy: {:3f}".format(accuracy_score(y_test, pred > 0.5)))

    #print("Confusion matrix:\n{}".format(confusion_matrix(y_test.argmax(axis=1), pred.argmax(axis=1))))
    print("Confusion matrix:\n{}".format(confusion_matrix(y_test.argmax(axis=1), pred.argmax(axis=1))))

    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)


if __name__ == '__main__':

    dataset = getDataset()
    #getmaxlen(dataset)
    vectorised_data = getcbow(dataset)
    print(type(vectorised_data))
    #print(vectorised_data)
    target = collect_expected(dataset) #Biased two classes {198, 2} lenght is 200
    #print (target)
    lstmModel(vectorised_data, target)
