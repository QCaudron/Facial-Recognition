# -*- coding: utf-8 -*-
"""

@author: matt
"""


# CNN examples : https://github.com/fchollet/keras/tree/master/examples
# CNN visualization : http://cs231n.github.io/understanding-cnn/

import os
import numpy as np
import pandas as pd
from keras import layers
from keras.models import Sequential
from matplotlib import pyplot

from keras.preprocessing.image import ImageDataGenerator, flip_axis
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
    
import CustomImageProcess as CIP
from sklearn.cross_validation import train_test_split



FTRAIN = '~/Python scripts/Facial recognition/training.csv'
FTEST = '~/Python scripts/Facial recognition/test.csv'

def load(test = False, col = None):
    fname = FTEST if test else FTRAIN
    df = pd.read_csv(fname)
    df.loc[:, 'Image'] = df.loc[:, 'Image'].apply(lambda im: np.fromstring(im, sep = ' '))
    if col:
        df = df.loc[:, list(col) + ['Image']]
#    print df.count()
    df = df.dropna()
    X = np.vstack(df['Image'].values) / 255.
    X = X.astype(np.float32)

    if not test:
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
#        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None
    return X, y
    
def load2d(test = False, col = None):
    X, y = load(col = col, test = test)
    X = X.reshape(-1, 1, 96, 96)
    return X, y
    
def tutorialConvolution():
        
    img_rows, img_cols = 96, 96 # images are 96x96 pixels
    img_channels = 1 # images are grey scale - if RGB use img_channels = 3
    nb_filter = 32 # common and efficient to use multiples of 2 for filters
    nb_epoch = 1
    batch_size = 32
    X, y = load2d()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)
    print X.shape
    model = Sequential()
    model.add(Convolution2D(nb_filter = nb_filter, nb_row = 3, nb_col = 3, 
#                            border_mode = 'same', # this causes a crash, even though default is 'same'....
                            input_shape = (img_channels, img_rows, img_cols))) # must set input shape for first call to Convolutional2D
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
    model.add(Dense(30))
    model.compile(loss = 'mean_squared_error', # other objectives : http://keras.io/objectives/
                  optimizer = 'Adam') # discussion : http://cs231n.github.io/neural-networks-3/ : "Adam is currently recommended as the default algorithm to use" - cs231n
    dataGen = createDataGen()
    model.fit_generator(dataGen.flow(X_train, y_train, batch_size=batch_size), nb_epoch = nb_epoch,
                                     samples_per_epoch = X_train.shape[0], validation_data = (X_test, y_test))
#    loss_and_metrics = model.evaluate(X_test, y_test, batch_size=32)
#    return loss_and_metrics


def plot_sample(x, y, axis):

    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    if y is not None:
        axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)
        


def createDataGen():
    return CIP.CustomImageProcess(featurewise_center=False,  # set input mean to 0 over the dataset
                                  samplewise_center=False,  # set each sample mean to 0
                                  featurewise_std_normalization=False,  # divide inputs by std of the dataset
                                  samplewise_std_normalization=False,  # divide each input by its std
                                  zca_whitening=False,  # apply ZCA whitening
                                  rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                                  width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
                                  height_shift_range=0,  # randomly shift images vertically (fraction of total height)
                                  horizontal_flip=True,  # randomly flip images
                                  vertical_flip=False)  # randomly flip images





def testMirror(i):
    
#    print('y:', y[i])
    datagen = createDataGen()
    X_shift, y_shift = datagen.random_transform(X[i], y[i])

#    print('New y:', y_shift)

    fig = pyplot.figure(figsize=(20, 15))
    ax = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
    plot_sample(X_shift, y_shift, ax)
    pyplot.show()


