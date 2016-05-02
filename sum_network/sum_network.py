#! /usr/bin/env python2

import tensorflow as tf
import numpy
from DataImport import *

# import the randomly generated training and testing data
training_data = TrainData()
test_data = TestData()

# first attempt will be just a softmax layer

x = tf.placeholder(tf.float32, [None, 4])

W = tf.Variable(tf.ones([4, 1024]))
b = tf.Variable(tf.zeros([1024]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 1024])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(.01).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(5000):
    batch_x, batch_y = training_data.next_batch(5000)
    train_accuracy = sess.run(accuracy, feed_dict={x:batch_x, y_:batch_y})
    print("step %d, training accuracy %g"%(i, train_accuracy))
    sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})

print(sess.run(accuracy, feed_dict={x: test_data.d.arrays, y_: test_data.d.labels}))
