# -*- coding: utf-8 -*-
"""

Documentation for this class : https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py

"""


from keras.preprocessing.image import *
import numpy as np

class CustomImageProcess(ImageDataGenerator):
    flip_indices = [
        (0, 2), (1, 3),
        (4, 8), (5, 9), (6, 10), (7, 11),
        (12, 16), (13, 17), (14, 18), (15, 19),
        (22, 24), (23, 25)]
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
#            x = self.standardize(x)
            bX[i] = x
            bY[i] = Y
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(bX[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}.{format}'.format(prefix=self.save_prefix,
                                                           index=current_index + i,
                                                           format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
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