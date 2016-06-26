from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

#input parameters- global variables
##fake data for tests
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')

#get mnist data
data_sets = input_data.read_data_sets(FLAGS.train_dir, FLAGS.fake_data)

#same as before? might not actually need
#batch_size = 50

#paceholders- 32 floaters
##IMAGE_PIXELS included from import mnist data above
images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                       mnist.IMAGE_PIXELS))
labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))

#tabs>spaces
##hiddenunits from variable/mnist
###weights/biases
def inference(images, hidden1_units, hidden2_units):

	#inputs hidden1
	with tf.name_scope('hidden1'):
		weights = tf.Variable(
    				tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                        stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
    					name='weights')
		biases = tf.Variable(tf.zeros([hidden1_units]),
                     name='biases')
	hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

	#inputs hidden2
	with tf.name_scope('hidden2'):
		weights = tf.Variable(
    				tf.truncated_normal([hidden1_units, hidden2_units],
                        stddev=1.0 / math.sqrt(float(hidden1_units))),
    					name='weights')
		biases = tf.Variable(tf.zeros([hidden2_units]),
                     name='biases')
	hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

	#softmax_linear outputs
	with tf.name_scope('softmax_linear'):
		weights = tf.Variable(
    				tf.truncated_normal([hidden2_units, NUM_CLASSES],
                        stddev=1.0 / math.sqrt(float(hidden2_units))),
    					name='weights')
		biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                     name='biases')
		logits = tf.nn.relutf.matmul(hidden2, weights) + biases
	return logits

#builds graph by adding required loss
##placeholders converted to 64
###cross entropy
####mean of cross entropy
def loss(logits, labels):
	labels = tf.to_int64(labels)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
		logits, labels, name='xentropy')
	loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
	return loss

#operations minimize loss by gradient descent
##scalar summ for snapshot loss
###descent optimizer with learning rate
####variable to track global step
def training(loss, learning_rate):
	tf.scalar_summary(loss.op.name, loss)
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	global_step = tf.Variable(0, name='global_step', trainable=False)
	train_op = optimizer.minimize(loss, global_step=global_step)
	return train_op

#evaluate logits at predicting the label
##returns a boolean on how many were predicted
def evaluation(logits, labels):
	correct = tf.nn.in_top_k(logits, labels, 1)
	return tf.reduce_sum(tf.cast(correct, tf.int32))
