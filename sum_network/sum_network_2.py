#! /usr/bin/env python2

import tensorflow as tf
import numpy
from DataImport import *

class DoublePAdam(tf.train.AdamOptimizer):
    def _valid_dtypes(self):
        return set([tf.float32, tf.float64])

# easy way of making variables for weight and bias
def weight_variable(shape):
    initial = tf.truncated_normal(shape, dtype=tf.float64, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, dtype=tf.float64, shape=shape)
    return tf.Variable(initial)

# function definitions for code cleanliness
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu=True)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')

# import the randomly generated training and testing data as bitwise lists
training_data = TrainData()
test_data = TestData()

x = tf.placeholder(tf.float64, shape=[None, 4], name="Placeholder_x")
y_ = tf.placeholder(tf.float64, [None, 1])

W = tf.Variable(tf.ones([4, 1], tf.float64))
#W = weight_variable([4, 1])
b = bias_variable([1])


y = (tf.matmul(x, W) + b)

cross_entropy = tf.abs(tf.sub(y, y_))

train_step = DoublePAdam(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(y, y_)
accuracy = tf.reduce_mean(tf.abs(tf.sub(y, y_)))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(1000000):
    batch_x, batch_y = training_data.next_batch(150)
    if i %100 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x:batch_x, y_:batch_y})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})

print(sess.run(accuracy, feed_dict={x: test_data.arrays, y_: test_data.labels}))

sess.close()
