"""

@author: matt
"""

# CNN examples : https://github.com/fchollet/keras/tree/master/examples
# Example of real time data augmentation : https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py

import os
import numpy as np
import pandas as pd
from keras import layers
from keras.models import Sequential
from matplotlib import pyplot

from keras.preprocessing.image import ImageDataGenerator, flip_axis
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
    
import CustomImageProcess as CIP
from sklearn.cross_validation import train_test_split

FTRAIN = '~/Python scripts/Facial recognition/training.csv'
FTEST = '~/Python scripts/Facial recognition/test.csv'

train_data = pd.read_csv(FTRAIN)
test_data = pd.read_csv(FTEST)

train_data.loc[:, 'Image'] = train_data.loc[:, 'Image'].apply(lambda im: np.fromstring(im, sep = ' '))
test_data.loc[:, 'Image'] = test_data.loc[:, 'Image'].apply(lambda im: np.fromstring(im, sep = ' '))

train_skinny = train_data.iloc[2284:].copy()
#print(train_skinny.shape, ', train_skinny shape before')
train_skinny.dropna(axis = 1, how = 'all', inplace = True)
#print(train_skinny.shape, ', train_skinny shape mid')
train_skinny.dropna(axis = 0, how = 'any', inplace = True)
#print(train_skinny.shape, ', train_skinny shape after')

train_fat = train_data.iloc[:2284].copy()
#print(train_fat.shape, ', train_fat shape before')
train_fat.dropna(axis = 0, how = 'any', inplace = True)
#print(train_fat.shape, ', train_fat shape after')

test_skinny = test_data.loc[test_data.ImageId >= 592].copy()
test_fat = test_data.loc[test_data.ImageId < 592].copy()

del train_data
del test_data

def load(df, test):
    X = np.vstack(df['Image'].values) / 255.
    X = X.astype(np.float32)
    X = X.reshape(-1, 1, 96, 96)
    if not test:
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        y = y.astype(np.float32)
    else:
        y = None
    return X, y
        
    
def Convolution(test, df, flip_indices):
    img_rows, img_cols = 96, 96 # images are 96x96 pixels
    img_channels = 1 # images are grey scale - if RGB use img_channels = 3
    nb_filter = 32 # common and efficient to use multiples of 2 for filters
    nb_epoch = 1
    batch_size = 32
    X, y = load(df, test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)
    
    model = Sequential()
    model.add(Convolution2D(nb_filter = nb_filter, nb_row = 3, nb_col = 3, 
#                            border_mode = 'same', # this causes a crash, even though default is 'same'....
                            input_shape = X_train.shape[1:])) # must set input shape for first call to Convolutional2D
    model.add(Activation('relu')) # default convolutional activation function - should try Leaky ReLU and PReLU functions : http://arxiv.org/abs/1502.01852
    model.add(MaxPooling2D()) # pooling 2x2 with stride 2 is default - reduces size of matrix by half in both dimensions
    
    # There is a new paper quesitoning the use of max pooling - finding that replacement with convolutional layers with increased stride works better    
    
    model.add(Dropout(0.1))
    model.add(Convolution2D(nb_filter = nb_filter * 2, nb_row = 2, nb_col = 2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Convolution2D(nb_filter = nb_filter * 4, nb_row = 2, nb_col = 2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.3))
    model.add(Flatten()) # flatten the data before fully connected layer (reg. neural net)
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    if len(flip_indices) == 12:
        model.add(Dense(30))
    else:
        model.add(Dense(8))
    model.compile(loss = 'mean_squared_error', # other objectives : http://keras.io/objectives/
                  optimizer = 'Adam') # discussion : http://cs231n.github.io/neural-networks-3/ : "Adam is currently recommended as the default algorithm to use" - cs231n
    dataGen = createDataGen(flip_indices = flip_indices, horizontal_flip = True)
    checkpointer = ModelCheckpoint(filepath="/tmp/weights.hdf5", verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor = 'val_loss', patience = 20, verbose = 0, mode = 'auto')
    model.fit_generator(dataGen.flow(X_train, y_train, batch_size = batch_size), nb_epoch = nb_epoch,
                                     samples_per_epoch = X_train.shape[0],
                                     validation_data = (X_test, y_test),
                                     callbacks = [checkpointer, earlystopping])
#    loss_and_metrics = model.evaluate(X_test, y_test, batch_size=32) 
#    return loss_and_metrics


def createDataGen(flip_indices, horizontal_flip):
    return CIP.CustomImageProcess(horizontal_flip = horizontal_flip,
                                  flip_indices = flip_indices)


flip_indices_fat = [(0, 2), (1, 3), (4, 8), (5, 9), (6, 10), (7, 11),
                    (12, 16), (13, 17), (14, 18), (15, 19), (22, 24), (23, 25)]
                                            
flip_indices_skinny = [(0, 2), (1, 3)]
                                            
#Convolution(test = False, df = train_fat, flip_indices = flip_indices_fat)
Convolution(test = False, df = train_skinny, flip_indices = flip_indices_skinny)



