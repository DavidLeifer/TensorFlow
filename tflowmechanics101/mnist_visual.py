#a small convolutional neural network
#download data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#Weight Initialization
##two functions-initialize them with a positive initial bias
###to avoid dead neurons
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#Convolution and Pooling
##stride size is 1, are 0 padded
##pooling is max over 2X2 blocks
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#First Convolutional Layer
##32 features for each 5X5 patch
#reshape x to a 4D tensor
#convolve x_image with the weight tensor
##add bias, apply the RELU function and max pool

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#Second convolutional layer
##stack several layers of this type
###64 features for 5X5 patch
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#Densely Conected Layer
##image size has been reduced to 7X7
###add fully connected layer with 1024 neaurons
####to allow processing on entire image
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#dropout- turn off/on during training
##solves overfitting problem on large networks
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Readout Layer- softmax regression
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#Train and evaluate the model
##identical to the simple one
###replaces steepest gradient with the ADAM optimizer
####ADAM controls the dropout rate
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

#for i in range(20000):
#  batch = mnist.train.next_batch(50)
#  if i%100 == 0:
#    train_accuracy = accuracy.eval(feed_dict={
#        x:batch[0], y_: batch[1], keep_prob: 1.0})
#    print("step %d, training accuracy %g"%(i, train_accuracy))
#  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

#glacially slow on CPU
#a little faster on GPU

merged_summary_op = tf.merge_all_summaries()
summary_writer = tf.train.SummaryWriter('/tmp/mnist_logs', sess.graph)
total_step = 0
while total_step < 2000:
  batch = mnist.train.next_batch(50)
  total_step += 1
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
  if total_step % 100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
      x:batch[0], y_: batch[1], keep_prob: 1.0
      })
    print("step %d, training accuracy %g"%(i, train_accuracy))
    summary_str = session.run(merged_summary_op)
    summary_writer.add_summary(summary_str, total_step)
    summary_writer.add_graph(sess.graph)

print("test accuracy %g"%accuracy.eval(feed_dict={
  x: mnist.test.images, y_: mnist.test.labels, keep_prob: 0.5}))

#run tensorboard
#python /usr/local/lib/python2.7/site-packages/tensorflow/tensorboard/tensorboard.py --logdir=/tmp/mnist_logs









