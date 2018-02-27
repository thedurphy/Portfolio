from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Convolution1D, MaxPooling1D, AtrousConvolution1D, RepeatVector
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.layers.wrappers import Bidirectional
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import *
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.initializers import *
from keras.utils import to_categorical
import datetime
import numpy as np

def normalize(x):
    return (np.array(x)-np.mean(x))/(np.std(x))

def shuffle_in_unison(a, b):
    # courtsey http://stackoverflow.com/users/190280/josh-bleecher-snyder
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b
 
def create_Xt_Yt(X, y, percentage=0.9):
    p = int(len(X) * percentage)
    X_train = X[0:p]
    Y_train = y[0:p]
     
    X_train, Y_train = shuffle_in_unison(X_train, Y_train)
 
    X_test = X[p:]
    Y_test = y[p:]

    return X_train, X_test, Y_train, Y_test


def remove_nan_examples(data):
    newX = []
    for i in range(len(data)):
        if np.isnan(data[i]).any() == False:
            newX.append(data[i])
    return newX

def pd2seq(dataframe, lookback, y_start = -1, normalize_col = None):
    idx = np.r_[range(dataframe.shape[1])]
    if normalize_col:
        idx = np.r_[tuple([i for i,j in enumerate(dataframe.columns) if j in normalize_col])]
    mat = dataframe.as_matrix()
    x, y = [], []
    for i in range(lookback, mat.shape[0]):
        y.append(mat[i,y_start:])
        x_temp = mat[i-lookback:i]
        x_temp[:, idx] = np.apply_along_axis(normalize, 0, x_temp[:, idx])
        x.append(x_temp)
    return np.array(x), np.array(y)



def seq_model(dataframe, lookback = 5, epoch = 100, batch_size = 128, y_start = -2, now = datetime.datetime.now().strftime('%Y.%m.%dT%H.%M.%S'), lstm = False,normalize_col = None, folder = 'trials'):
    '''
    dataframe should contain the final input variables with the output variable in the last column
    dataframe : pandas.DataFrame object
    lookback : how far back, sequantially, the features will be for each outback (default = 5)
    epoch : how many epochs (default = 100)
    batch_size : how large each batch will be for weight updates (default = 128)
    y_start : this is to accomodate classifications greater than 2.  The number should represent the starting column index
    of the outcome variable. (default = -2)
    now = current time to indicate the beginning of the modeling and to name file.
    '''
    filepath="{}/forexrnn_{}.weights".format(folder, now)
    x, y = pd2seq(dataframe, lookback, y_start, normalize_col)
    X_train, X_test, Y_train, Y_test = create_Xt_Yt(x, y)
    EMB_SIZE = dataframe.shape[1]
    model = Sequential()
    model.add(Convolution1D(input_shape = (lookback, EMB_SIZE),
                            filters=32,
                            kernel_size=4,
                            padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))

    model.add(Convolution1D(filters=16,
                            kernel_size=4,
                            padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))

    model.add(Convolution1D(filters=8,
                            kernel_size=4,
                            padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))

    if lstm == True:
        model.add(LSTM(12, return_sequences=True))
        model.add(LSTM(6, return_sequences=True))

    model.add(Flatten())

    model.add(Dense(80))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Dense(40))
    model.add(BatchNormalization())
    model.add(LeakyReLU())                

    model.add(Dense(20))
    model.add(BatchNormalization())
    model.add(LeakyReLU())                

    model.add(Dense(y.shape[1]))
    model.add(Activation('softmax'))

    opt = Nadam(lr=0.002)

    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=30, min_lr=0.000001, verbose=1)
    checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)


    model.compile(optimizer=opt, 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, Y_train, 
              epochs = epoch, 
              batch_size = batch_size, 
              verbose=1, 
              validation_data=(X_test, Y_test),
              callbacks=[reduce_lr, checkpointer],
              shuffle=True)
    model.load_weights(filepath)
    pred = model.predict(np.array(X_test))
    C = confusion_matrix([np.argmax(y) for y in Y_test], [np.argmax(y) for y in pred])
    avg_acc = np.amax((C/C.astype(np.float).sum(axis=1)), axis = 1).mean()
    acc = (np.argmax(pred, axis = 1)==np.argmax(Y_test, axis = 1)).mean()
    model.save('{}/forexrnn_{}_{}.model'.format(folder,now, acc))
    return model, pred, C, avg_acc, Y_test, x, y

def lstm_model(dataframe, lookback = 5, epoch = 100, batch_size = 128, y_start = -2, now = datetime.datetime.now().strftime('%Y.%m.%dT%H.%M.%S'), normalize_col = None, folder = 'trials'):
    '''
    dataframe should contain the final input variables with the output variable in the last column
    dataframe : pandas.DataFrame object
    lookback : how far back, sequantially, the features will be for each outback (default = 5)
    epoch : how many epochs (default = 100)
    batch_size : how large each batch will be for weight updates (default = 128)
    y_start : this is to accomodate classifications greater than 2.  The number should represent the starting column index
    of the outcome variable. (default = -2)
    now = current time to indicate the beginning of the modeling and to name file.
    '''
    filepath="{}/forexrnn_{}.weights".format(folder, now)
    x, y = pd2seq(dataframe, lookback, y_start, normalize_col)
    X_train, X_test, Y_train, Y_test = create_Xt_Yt(x, y)
    EMB_SIZE = dataframe.shape[1]
    model = Sequential()
    model.add(Convolution1D(input_shape = (lookback, EMB_SIZE),
                            filters=32,
                            kernel_size=4,
                            padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))

    model.add(Convolution1D(filters=16,
                            kernel_size=4,
                            padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))

    model.add(Convolution1D(filters=8,
                            kernel_size=4,
                            padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(80))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Dense(40))
    model.add(BatchNormalization())
    model.add(LeakyReLU())                

    model.add(Dense(20))
    model.add(BatchNormalization())
    model.add(LeakyReLU())                

    model.add(Dense(y.shape[1]))
    model.add(Activation('softmax'))

    opt = Nadam(lr=0.002)

    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=30, min_lr=0.000001, verbose=1)
    checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)


    model.compile(optimizer=opt, 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, Y_train, 
              epochs = epoch, 
              batch_size = batch_size, 
              verbose=1, 
              validation_data=(X_test, Y_test),
              callbacks=[reduce_lr, checkpointer],
              shuffle=True)
    model.load_weights(filepath)
    pred = model.predict(np.array(X_test))
    C = confusion_matrix([np.argmax(y) for y in Y_test], [np.argmax(y) for y in pred])
    avg_acc = np.amax((C/C.astype(np.float).sum(axis=1)), axis = 1).mean()
    acc = (np.argmax(pred, axis = 1)==np.argmax(Y_test, axis = 1)).mean()
    model.save('{}/forexrnn_{}_{}.model'.format(folder,now, acc))
    return model, pred, C, avg_acc, Y_test, x, y