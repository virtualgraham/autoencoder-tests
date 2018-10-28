import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', validation_size=0)

img = mnist.train.images[2]
print('img.shape', img.shape)
plt.imshow(img.reshape((28, 28)), cmap='Greys_r')
plt.show()