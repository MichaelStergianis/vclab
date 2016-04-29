#! /usr/bin/env python2

import tensorflow as tf
import numpy as np
## import cv2 if you have opencv and want to see the images
# import cv2

#### sample code for displaying each image
### for i in images:
###     cv2.imshow('image', np.reshape(i, (28, 28), order='C'))
###     cv2.waitKey(0)
###     cv2.destroyAllWindows()

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# import the array of images, each image is shape=(1,784)
images = mnist.train.images


# initialize our variables and placeholders
x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# softmax on W*x + b
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# training step
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# initialize our variables
init = tf.initialize_all_variables()

# create a session and run
session = tf.Session()
session.run(init)

# run the training step 1000 times
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    session.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(session.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
