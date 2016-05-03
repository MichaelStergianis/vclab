#! /usr/bin/env python2

import tensorflow as tf
import numpy
from DataImport import *

# easy way of making variables for weight and bias
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# function definitions for code cleanliness
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')

# import the randomly generated training and testing data as bitwise lists
training_data = TrainData(bitwise=True)
test_data = TestData(bitwise=True)

x = tf.placeholder(tf.float32, shape=[None, 32], name="Placeholder_x")
y_ = tf.placeholder(tf.float32, [None, 1021])

####################### FIRST LAYER ###############################
####################################################################

# initialize our variables
W_conv1 = weight_variable([2, 4, 1, 32])
b_conv1 = bias_variable([32])

# reshape x to a 4-d tensor
x_mat = tf.reshape(x, [-1, 4, 8, 1])

h_conv1 = tf.nn.relu(conv2d(x_mat, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

####################################################################


####################### SECOND LAYER ###############################
####################################################################

W_conv2 = weight_variable([2, 4, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

####################################################################

####################### THIRD LAYER ################################
####################################################################

W_conv3 = weight_variable([2, 4, 64, 128])
b_conv3 = bias_variable([128])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

####################################################################

####################### DENSE LAYER ################################
####################################################################

W_fc1 = weight_variable([4 * 8 * 128, 1024])
b_fc1 = bias_variable([1024])

h_pool3_flat = tf.reshape(h_pool3, [-1, 4*8*128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

# Dropout layer
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

####################################################################

####################### READOUT LAYER ##############################
####################################################################

W_fc2 = weight_variable([1024, 1021])
b_fc2 = bias_variable([1021])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

####################################################################

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(10001):
    batch_x, batch_y = training_data.next_batch(1000)
    if i %100 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x:batch_x, y_:batch_y, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    sess.run(train_step, feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})

print(sess.run(accuracy, feed_dict={x: test_data.arrays, y_: test_data.labels, keep_prob: 1.0}))

sess.close()
