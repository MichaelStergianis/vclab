#! /usr/bin/env python2
import tensorflow as tf
import load_mat as lm
import time

from tensorflow.examples.tutorials.mnist import input_data

# Many aspects  of this tutorial will be the same as the naive tutorial. 
# The reason is, while we want to change the architecture of our network
# we may not want to change things like cross entropy, or the fact that we use
# weights and biases.

# our definition of what a weight variable is, we supply the shape and it makes
# a gaussian distribution across that shape with a std deviation of .1
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# our definition of bias variables, we initialize it to be 0.1 in all dimensions
# to avoid dead connections
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# convolution is a fairly simple operation but it won't be explained in this
# tutorial
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# as with convolution max pooling will not be explained here
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], 
                          strides=[1, 2, 2, 1], padding='SAME')

# matlab data as opposed to tensorflow mnist
data = lm.load_mat("ex4data1.mat")

# optionally you can use tensorflows mnist data as well
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# import the array of images, each image is shape=(1,784)
images = mnist.train.images

#### sample code for displaying each image
### for i in images:
###     cv2.imshow('image', np.reshape(i, (28, 28), order='C'))
###     cv2.waitKey(0)
###     cv2.destroyAllWindows()

# initialize placeholders same as last tutorial
x = tf.placeholder(tf.float32, [None, 400])
y_ = tf.placeholder(tf.float32, [None, 10])

######################## FIRST LAYER #############################
# Our first layer is going to perform a 2-d convolution over our raw image
# Then it is going to perform some max pooling, which will reduce the output
# for the next layer
##################################################################

# our weight variable has 
# 5x5 window for convolution
# 1 channel of input because this is a grayscale picture
# 32 channels of output because we want to use 32 different filters
W_conv1 = weight_variable([5, 5, 1, 32])
# our bias variables shape must match the number of output channels
b_conv1 = bias_variable([32])

# reshape x to 4d tensor so that we can convolve over the image
x_image = tf.reshape(x, [-1, 20, 20, 1])

# convolution and max pool
# relu is one of the activation funcitons for a neuron
# it is describes as such relu(x) = max(0, x) nice and simple
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

##################################################################

######################## SECOND LAYER ############################
# The second layer is much like the first, but taking in 32 input channels
# and outputting 64 channels
##################################################################

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

##################################################################

######################## DENSE LAYER #############################
# So now we will use a fully connected layer, ie a matrix multiplication
# across a variety of our weights
##################################################################

# max pooling reduces the dimensions of our tensor
# 28x28 / 2x2 = 14x14 1st time
# 14x14 / 2x2 = 7x7   2nd time
# 7x7x64 output channels is the number of values in this tensor
W_fc1 = weight_variable([5 * 5 * 64, 1024])
b_fc1 = bias_variable([1024])

# now we want to flatten h_pool2 so that it is not a 4d tensor, but a 2d tensor
h_pool2_flat = tf.reshape(h_pool2, [-1, 5*5*64])
# we then feed the matrix multiplication to a relu layer of neurons
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

##################################################################

######################## DROPOUT LAYER ###########################
# Dropout is quite an important notion in neural networks
# Dropout is the process of randomly selecting neurons to ignore
# usually during the training process. This is done to reduce learning
# the training set, instead of the actual data
##################################################################

# we make our probability to keep a given neuron a placeholder so that we
# can feed it a value
keep_prob = tf.placeholder(tf.float32)
# This function facilitates dropout, given a scalar and a tensor it will apply
# the operation element wise
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

##################################################################

######################## READOUT LAYER ###########################
# The readout layer is effectively our naive mnist method
# it does a matrix multiplication of our image tensor and a wegihts tensor
# and applies a softmax regularization on that data so that we have probabilities
# between [0,1] adding to 1
##################################################################

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

##################################################################

# same cross entropy as our last example, common for a logistic regression
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))

# training step
# in this example we are using an adam optimizer, a free pdf publication can be
# found on the tensorflow api documentation for this optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# create a session and run
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.initialize_all_variables())

# run the training step 20000 times
for i in range(20000):
    with sess.as_default(): 
        batch_xs, batch_ys = data.next_batch(50)
        if i % 100 == 0:
            # in testing we use a keep prob of 1.0 to test all neurons
            train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_:batch_ys, keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        # when training we use a keep probability of 0.5, this is a common
        # probability for dropout, and for most cases is close to optimal
        train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})


print("test accuracy %g"%accuracy.eval(session=sess, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
sess.close()
