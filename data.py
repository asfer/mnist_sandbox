import array
import struct

import numpy as np


class MNIST:
    """ MNIST dataset is composed of digit images of size 28x28 and its labels """

    def __init__(self, data_dir):
        self.train_data, self.train_labels = self.parse_images(data_dir + '/train-images-idx3-ubyte'),  \
                                             self.parse_labels(data_dir + '/train-labels-idx1-ubyte')
        self.test_data, self.test_labels = self.parse_images(data_dir + '/t10k-images-idx3-ubyte'), \
                                           self.parse_labels(data_dir + '/t10k-labels-idx1-ubyte')

    @staticmethod
    def parse_images(filename):
        with open(filename, 'rb') as f:
            magic, items, rows, cols = struct.unpack('>IIII', f.read(16))
            assert magic == 2051
            size = rows * cols
            images = array.array('B', f.read())
            assert items * size == len(images)
            return np.array(images, dtype=np.int8).reshape((items, size), order='C')

    @staticmethod
    def parse_labels(filename):
        with open(filename, 'rb') as f:
            magic, items = struct.unpack('>II', f.read(8))
            assert magic == 2049
            labels = array.array('B', f.read())
            assert len(labels) == items
            return np.array(labels, dtype=np.int8).reshape((items, 1))


mnist = MNIST('./data')
print(mnist.train_data.shape, mnist.train_labels.shape)
