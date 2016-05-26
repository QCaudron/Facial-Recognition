"""

Documentation for this class : https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py

"""


from keras.preprocessing.image import *
import numpy as np

class CustomImageProcess(ImageDataGenerator):
    
    def __init__(self, flip_indices, horizontal_flip):
        ImageDataGenerator.__init__(self, horizontal_flip)
        self.flip_indices = flip_indices    
    
    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see # http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.flow_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        bX = np.zeros(tuple([current_batch_size] + list(self.X.shape)[1:]))
        bY = np.zeros(tuple([current_batch_size] + list(self.y.shape)[1:]))

        for i, j in enumerate(index_array):
            x = self.X[j]
            Y = self.y[j]
            x, Y = self.random_transform(x.astype('float32'), Y.astype('float32'))
            bX[i] = x
            bY[i] = Y
            
        return (bX, bY)
        
    def random_transform(self, x, Y):
        # x is a single image, so it doesn't have image number at index 0
        img_row_index = self.row_index - 1
        img_col_index = self.col_index - 1
        img_channel_index = self.channel_index - 1
#        print('Original Y:', Y)


        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_index)
                Y[::2] *= -1
                for a, b in self.flip_indices:
#                    print a, 'swaps with', b
                    Y[a], Y[b] = Y[b], Y[a]
        return x, Y
