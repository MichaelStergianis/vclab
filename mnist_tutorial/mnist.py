import numpy as np
import cv2

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# import the array of images, each image is shape=(1,784)
images = mnist.train.images

#### sample code for displaying each image
### for i in images:
###     cv2.imshow('image', np.reshape(i, (28, 28), order='C'))
###     cv2.waitKey(0)
###     cv2.destroyAllWindows()


