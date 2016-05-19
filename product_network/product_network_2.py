#! /usr/bin/env python2

import tensorflow as tf
import numpy
from DataImport import *

class DoublePAdam(tf.train.AdamOptimizer):
    def _valid_dtypes(self):
        return set([tf.float32, tf.float32])

# easy way of making variables for weight and bias
def weight_variable(shape):
    initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, dtype=tf.float32, shape=shape)
    return tf.Variable(initial)

# function definitions for code cleanliness
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu=True)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')

# import the randomly generated training and testing data as bitwise lists
##training_data = AdvancedTrainData()
##test_data = AdvancedTestData()
training_data = TrainData()
test_data = TestData()

x = tf.placeholder(tf.float32, shape=[None, 4], name="Placeholder_x")
y_ = tf.placeholder(tf.float32, [None, 1])

#W = tf.Variable(tf.ones([36, 1]))
#
#b = bias_variable([1])
#
#y = tf.matmul(x, W) + b

layer_1 = tf.contrib.layers.fully_connected(x, 256, activation_fn=tf.tanh)

layer_2 = tf.contrib.layers.fully_connected(layer_1, 256, activation_fn=tf.tanh)

layer_3 = tf.contrib.layers.fully_connected(layer_2, 256, activation_fn=tf.nn.relu)
#
#layer_4 = tf.contrib.layers.fully_connected(layer_3, 1024, activation_fn=tf.nn.relu)
#
#layer_5 = tf.contrib.layers.fully_connected(layer_4, 1024, activation_fn=tf.nn.relu)

y = tf.contrib.layers.fully_connected(layer_3, 1, activation_fn=tf.nn.relu)

cross_entropy = (tf.square(tf.sub(y, y_)))

train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

train_step_2 = tf.train.AdadeltaOptimizer().minimize(cross_entropy)

correct_prediction = tf.equal(y, y_)
accuracy = tf.abs(tf.sub(y, y_))
reduced_accuracy_max = tf.reduce_max(tf.abs(tf.sub(y, y_)))
reduced_accuracy_mean = tf.reduce_mean(tf.abs(tf.sub(y, y_)))
reduced_accuracy_min = tf.reduce_min(tf.abs(tf.sub(y, y_)))


sess = tf.Session()
sess.run(tf.initialize_all_variables())

i = 0
while True:
    batch_x, batch_y = training_data.next_batch(100)
    i += 1
    if i %100 == 0:
        #train_accuracy = sess.run(accuracy, feed_dict={x:batch_x, y_:batch_y})
        train_reduced_max = sess.run(reduced_accuracy_max, feed_dict={x:batch_x, y_:batch_y})
        train_reduced_mean = sess.run(reduced_accuracy_mean, feed_dict={x:batch_x, y_:batch_y})
        train_reduced_min = sess.run(reduced_accuracy_min, feed_dict={x:batch_x, y_:batch_y})
        #print("step %d, training accuracy "%(i), train_accuracy)
        print("step {0:06d}, training accuracy max: {1:05.7f} mean: {2:05.7f} min: {3:05.7f}".format(i, 
            train_reduced_max, train_reduced_mean, train_reduced_min))
    sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})

for i in range(100000):
    batch_x, batch_y = training_data.next_batch(100)
    if i %100 == 0:
        #train_accuracy = sess.run(accuracy, feed_dict={x:batch_x, y_:batch_y})
        #train_reduced_accuracy = sess.run(reduced_accuracy, feed_dict={x:batch_x, y_:batch_y})
        train_reduced_max = sess.run(reduced_accuracy_max, feed_dict={x:batch_x, y_:batch_y})
        train_reduced_mean = sess.run(reduced_accuracy_mean, feed_dict={x:batch_x, y_:batch_y})
        train_reduced_min = sess.run(reduced_accuracy_min, feed_dict={x:batch_x, y_:batch_y})
        #print("step %d, training accuracy "%(i), train_accuracy)
        print("step {0:06d}, training accuracy max: {1:5.7f} mean: {2:5.7f} min: {3:5.7f}".format(i, 
            train_reduced_max, train_reduced_mean, train_reduced_min))
    sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})

test_accuracy = sess.run(accuracy, feed_dict={x: test_data.arrays, y_: test_data.labels})
print("test accuracy", test_accuracy)

sess.close()
