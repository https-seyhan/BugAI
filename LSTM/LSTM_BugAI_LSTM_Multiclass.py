import numpy as np
import datetime
import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Embedding, LSTM
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from random import shuffle, sample
from gensim.models import Word2Vec
from random import shuffle, sample
from keras.utils import to_categorical

# The objective of this model is to predict number of bugs in a given AST code using LSTM
# This is a multiclass (10 classes classification)

traindata = '/home/saul/deeplearning/aclImdb/train'

maxlen = 5 #400 # number of words in a row
batch_size = 32 #32
embedding_dims = 300 #300 #5 #300lstmlayer
epochs = 50 #best result is with 20
num_neurons = 250 #50
sample_size =  998 #10 # The training samples are relatively few. Use pre-trained samples to increase the accuracy.

def getcbow(dataset):
    sentences = []
    vectorised_codes = []
    ast = [row.split('::') for row in dataset['classname']]
    #the input to the cbow is list of list of each line
    #size of the word vector of a given token must be equal to embedding_dim of the LSTM model
    cbowmodel = Word2Vec(ast, min_count=1, size= embedding_dims, workers=3, window=3, sg=0)
    #print(ast[:2])
    print (' CBOW model ', cbowmodel)
    
    #Test cbow model
    print("Test CBOW on the data")
    print(cbowmodel['eclipse'])
    
    classes = dataset['classname']

    for codes in classes:

        linecode = []
        tokens = codes.split('::')
        #print(tokens)
        sentences.append(tokens)
        for token in tokens:
            try:
                #print("Token ", token)
                #linecode.append(token)
                #print("Word Vector ", len(cbowmodel[token]))
                linecode.append(cbowmodel[token])
            except KeyError:
                pass
        vectorised_codes. append(linecode)
    #print(len(linecode))
    #print(linecode)


    #print('Line codes ', linecode)
    #print('Vectorised Codes ', vectorised_codes[0])
    #print('Vectorised Codes ', len(vectorised_codes))
    #print(f'Sentences: {sentences}')

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

    split_point =  int(len(vectorised_data) * .8)
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
    #print(type(x_train))

    #y_train = np.array(y_train)

    x_train = np.reshape(x_train, (len(x_train), maxlen, embedding_dims))
    print("X_TRAIN Reshape Completed ")
    #y_train = np.array(y_train)
    y_train = to_categorical(y_train, 10)


    x_test = np.reshape(x_test, (len(x_test), maxlen, embedding_dims))
    #y_test = np.array(y_test)
    y_test = to_categorical(y_test, 10) #convert classes into categorical values.
    print("Data Reshape Ended ")
    
    model = Sequential()
    #model.add(Embedding(embedding_dims, batch_size))

    model.add(LSTM(num_neurons, return_sequences=True, input_shape=(maxlen, embedding_dims)))
    model.add(Dense(32, activation='relu'))
    #model.add(LSTM(num_neurons, return_sequences=True, input_shape=(maxlen / 2, embedding_dims)))
    model.add(LSTM(num_neurons, return_sequences=True, input_shape=(maxlen , embedding_dims))) #stack LSTMs

    model.add(Dropout(.2))
    model.add(Flatten()) # dense layer expects a flat vectors of n elements
    model.add(Dense(10, activation='relu')) # one class
    #model.add(Dense(1, activation='relu'))  # one class model.add(Dense(1, activation='relu')) # one class
    #model.add(Dense(1, activation='sigmoid'))  # one class
    model.add((Dense(10))) # add ten classes
    model.add(Activation('softmax'))  # one class
    #model.compile('rmsprop', 'binary_crossentropy', metrics=['accuracy'])
    #model.compile('rmsprop', 'binary_crossentropy', metrics=['accuracy'])
    model.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    fitmodel(model, x_train, y_train, x_test, y_test, batch_size, epochs)

def fitmodel(model, x_train, y_train, x_test, y_test, batch_size, epochs):
    # print('X Train ', x_train.shape)
    # print(' Y Train ', y_train.shape)

    # print('X Test ', x_test.shape)
    # print('Y Test ', y_test.shape)

    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))

    # model.fit(np.array(x_train), np.array(y_train),
    # batch_size = batch_size,
    # epochs =epochs,
    # validation_data=(np.array(x_test), np.array(y_test)))

    model_structure = model.to_json()
    with open("lstm_model.json", "w") as json_file:
        json_file.write(model_structure)

    plotresults(history)
    # model.save_weights("rnn_weights.h5)

    # print(fittedmodel)

def collect_expected(dataset):
    expected = []
    #bugsdata = pd.read_csv('bug-metrics.csv', sep= ',')
    #print(dataset.columns)

    bugs = dataset['bugs']
    #print(bugs)
    for bug in bugs:
        #print(bug)
        expected.append(bug)

    return expected

def getDataset():

    dataset = pd.read_csv('bug-metrics.csv', sep= ',')

    #keep = ['classname', 'bugs']
    dataset = dataset.sample(n= sample_size, replace= True, random_state=1)
    #print(dataset.head(5))
    #print(len(dataset))
    #dataset = sample(dataset, 100)
    #shuffle(dataset)
    dataset.to_csv('sampledataset.csv')
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
        

def plotresults(history):
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


if __name__ == '__main__':

    dataset = getDataset()
    #getmaxlen(dataset)
    vectorised_data = getcbow(dataset)
    print(type(vectorised_data))
    #print(vectorised_data)
    target = collect_expected(dataset)
    #print (target)
    lstmModel(vectorised_data, target)
    
