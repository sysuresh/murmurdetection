import h5py
import numpy as np
import random
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pandas as pd
from keras import backend as K
from scipy.signal import decimate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPool1D, GlobalAvgPool1D, Dropout, BatchNormalization, Dense, LSTM
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.utils import np_utils
from keras.regularizers import l2

np.random.seed(1729)
random.seed(13)
INPUT_LIB=''
CLASSES = ['artifact', 'normal', 'murmur']
CODE_BOOK = {x:i for i,x in enumerate(CLASSES)}   
NB_CLASSES = len(CLASSES)

def fbeta(y_true, y_pred, threshold_shift=0):
    beta = 2

    y_pred = K.clip(y_pred, 0, 1)
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin), axis=1) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)), axis=1)
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)), axis=1)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return K.mean((beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon()))   

def clean_filename(fname, string):   
    filename = fname.split('/')[1]
    if filename[:2] == '__':        
        return (string + filename)
    return filename

def load_wav_file(name, path):
    a, b = wavfile.read(path + name)
    return b

def repeat_to_length(arr, length):
    result = np.empty((length,), dtype = 'float32')
    pos = 0
    while pos + len(arr) <= length:
        result[pos:pos+len(arr)] = arr
        pos += len(arr)
    if pos < length:
        result[pos:length] = arr[:length-pos]
    return result

df = pd.read_csv('set_a.csv')
df['fname'] = df['fname'].apply(clean_filename, string='Aunlabelledtest')
df['label'].fillna('unclassified')
df['time_series'] = df['fname'].apply(load_wav_file, path=INPUT_LIB + 'set_a/')    
df['len_series'] = df['time_series'].apply(len)
MAX_LEN = max(df['len_series'])
print(MAX_LEN)
df['time_series'] = df['time_series'].apply(repeat_to_length, length=MAX_LEN) 

X = np.stack(df['time_series'].values, axis=0)
print(X)
new_labels =[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 1, 1, 2, 1, 
             2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
             2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 
             1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 0, 2, 2, 1, 1, 1, 1, 1, 
             0, 1, 0, 1, 1, 1, 2, 1, 0, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 
             1, 0, 0, 0, 0, 0, 0, 1, 0, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

new_labels = np.array(new_labels, dtype='int')
y = np_utils.to_categorical(new_labels)

X_train, X_test, y_train, y_test, train_filenames, test_filenames = train_test_split(X, y, df['fname'].values, test_size=0.25)
X_train = decimate(X_train, 8, axis=1, zero_phase=True)
X_train = decimate(X_train, 8, axis=1, zero_phase=True)
X_train = decimate(X_train, 4, axis=1, zero_phase=True)
X_test = decimate(X_test, 8, axis=1, zero_phase=True)
X_test = decimate(X_test, 8, axis=1, zero_phase=True)
X_test = decimate(X_test, 4, axis=1, zero_phase=True)
X_train = scale(X_train, axis=1,with_mean=True)
X_test = scale(X_test, axis=1,with_mean=True)

X_train = X_train.reshape(*X_train.shape, 1)
X_test = X_test.reshape(*X_test.shape, 1)
model = Sequential()
model.add(Conv1D(filters=4, kernel_size=9, activation='relu',input_shape = X_train.shape[1:],kernel_regularizer = l2(0.025)))
model.add(MaxPool1D(strides=4))
model.add(BatchNormalization())
model.add(Conv1D(filters=4, kernel_size=9, activation='relu',kernel_regularizer = l2(0.05)))
model.add(MaxPool1D(strides=4))
model.add(BatchNormalization())
model.add(Conv1D(filters=32, kernel_size=9, activation='relu', kernel_regularizer = l2(0.05)))
model.add(MaxPool1D(strides=4))
model.add(BatchNormalization())
model.add(Conv1D(filters=64, kernel_size=9, activation='relu', kernel_regularizer = l2(0.05)))
model.add(MaxPool1D(strides=4))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(LSTM(20, return_sequences=True))
model.add(LSTM(10))
model.add(Dense(3, activation='softmax'))
def get_batch(X_train, y_train, size):
    while 1:
        idx = random.sample([*range(len(y_train))], size)
        X_batch, y_batch = X_train[idx], y_train[idx]
        X_batch = X_batch.reshape(*X_batch.shape[:-1])
        colstarts = np.random.randint(0, X_batch.shape[1], X_batch.shape[0])
        shiftindices = np.mod(colstarts.reshape(*colstarts.shape, 1) + np.arange(X_train.shape[1]), X_batch.shape[1])
        X_batch = X_batch[np.arange(X_batch.shape[0])[:,None], shiftindices]
        yield X_batch.reshape(*X_batch.shape, 1) , y_batch

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', fbeta])
hist = model.fit_generator(get_batch(X_train, y_train, 8),epochs=40, steps_per_epoch=1000,validation_data=(X_test, y_test), 
        callbacks=[ModelCheckpoint("model.h5", monitor="val_fbeta", save_best_only=True), LearningRateScheduler(lambda epoch: 1e-2 * 0.8**epoch)],verbose=2)
