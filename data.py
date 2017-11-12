import array
import struct

import numpy as np
from PIL import Image


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
            assert magic == 2051 and rows == 28 and cols == 28
            images = array.array('B', f.read())
            assert items * rows * cols == len(images)
            return np.array(images, dtype=np.int8).reshape((items, cols, rows), order='C')

    @staticmethod
    def parse_labels(filename):
        with open(filename, 'rb') as f:
            magic, items = struct.unpack('>II', f.read(8))
            assert magic == 2049
            labels = array.array('B', f.read())
            assert len(labels) == items
            return np.array(labels, dtype=np.int8).reshape((items, 1))

    @staticmethod
    def display(array):
        image = Image.fromarray(array)
        scaled_shape = tuple([8 * i for i in array.shape])
        image = image.resize(scaled_shape)
        image.show()


mnist = MNIST('./data')


if __name__ == '__main__':
    example = mnist.train_data[41, :, :]
    print(example.shape)
    mnist.display(example)
