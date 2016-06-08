#! /usr/bin/env python2
import tensorflow as tf
import load_mat as lm
import time

start = time.time()

# Small class to facilitate getting data
data = lm.load_mat("ex4data1.mat")

# next batch is going to be the main function we use
# but for now let's describe our graph

# our images in this case are 20x20 so 400 length vectors
x  = tf.placeholder(tf.float32, shape=[None, 400])

# our correct labels will also be held in a placeholder
# it consists of 10 floating point numbers each indicating a probability
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# now for our variables, this will just be a one layer network, so we are going
# to have 1 weights variable and 1 bias variable
W = tf.Variable(tf.zeros([400, 10]), tf.float32)
b = tf.Variable(tf.zeros([10], tf.float32))

# softmax is a normalized exponential
# it will be described in another document in this folder in more detail called softmax.md
# basically it will take the values these matrix multiplications output
# and normalize them between 0 and 1 as floats
y = tf.nn.softmax(tf.matmul(x, W) + b)

# now it's time to describe our loss function we will call it cross_entropy
loss = tf.mul(y_, tf.log(y))
# two things went on there one is tf.log(y)
## this will do an element wise log on y, ie 
### for i in y: i = log(y)
## the second is tf.mul(y_, tf.log(y))
### this does, you guessed it, element wise multiplication of y_ and log(y)
# for brevity the reason we do this will be described in another file called cross_entropy.md

# reduce sum will in this case move across the tensor and accumulate a sum
# across the values of the tensor and return that sum, basically:
# sum = 0
# for i in tensor:
#     sum += i
# return sum
cross_entropy = -tf.reduce_sum(loss)

# this step is fairly easy, tensorflow has some built in optimizers
# we're going to use the Gradient Descent Optimizer for this one
# each optimizer has the built in function minimize, basically minimize our loss
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# tf.argmax returns the index of the largest elemnt in our tensor
# tf.equal returns a True false array
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# tf.reduce mean will do something similar to sum, but at the final step it will
# divide sum by the number of elements in the tensor
# basically we're just getting an average of how many we got right
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Now, because we're describing a graph not creating variables, we need to tell
# tensorflow that IT needs to create variables
init = tf.initialize_all_variables()

# now we create a tensorflow session, basically this tells our c backend to start
# creating variables and gives us our bridge, sess, to our execution environment
sess = tf.Session() # also tf.InteractiveSession()
sess.run(init)

# define our batch size for training, not necessary here, but for clarity
batch_size = 75
# now we're going to train let's say 1000 times
for i in range(1000):
    # this is our function built into data, basically it will return 
    # batch_size random elements from our data set as a tuple
    batch_x, batch_y = data.next_batch(batch_size)
    sess.run(train_step, feed_dict={x:batch_x, y_:batch_y})
    if i%100 == 0:
        print(sess.run(accuracy, feed_dict={x:batch_x, y_:batch_y}))

# Now we're going to test this network 
xs, ys = data.get_data()
print("Total Accuracy: " + str(sess.run(accuracy, feed_dict={x:xs, y_:ys})))

print("Total time to run " + str(time.time() - start) + "s")

###############################################################################
# All of this took about 30 lines of code. Everything else is either comments
# or white space. That being said, it's not a great network. Even when we have
# separate test data this network only yields around 92% test accuracy. For
# mnist that is astonishingly bad, and current state of the art as of 
# May 19 2016 is ~ 99.79% accurate or a 0.21% error rate
###############################################################################

