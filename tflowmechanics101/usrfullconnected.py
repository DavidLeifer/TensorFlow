from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

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

#same old placeholders
def placeholder_inputs(batch_size):
	images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         mnist.IMAGE_PIXELS))
	labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
  	return images_placeholder, labels_placeholder

#fills feed_dict gap
##feed_dict maps placeholders to values
def fill_feed_dict(data_set, images_pl, labels_pl):
	images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
													FLAGS.fake_data)
	feed_dict = {
		images_pl: images_feed,
		labels_pl: labels_feed,
	}
	return feed_dict

#runs eval against unit in time (epoch)
##sess of trained model, eval_correct # of correct predictions
def do_eval(
	sess, 
	eval_correct,
	images_placeholder,
	labels_placeholder,
	data_set):

#one epoch of eval, counts number of correct predictions
	true_count = 0
	steps_per_epoch = data_set.numexamples // FLAGS.batch_size
	num_examples = steps_per_epoch * FLAGS.batch_size

#training loop
	for step in xrange(steps_per_epoch):
		feed_dict = fill_feed_dict(
			data_set,
			images_placeholder,
			labels_placeholder)
		true_count += sess.run(eval_correct, feed_dict=feed_dict)
		precision = true_count / num_examples
		print(' num examples: %d num correct: %d Precisions @ 1: %.04f' %
			(num_examples, true_count, precision))

#function trains mnist data- looks familiar
def run_training():
	data_sets = input_data.read_data_sets(FLAGS.train_dir, FLAGS.fake_data)

	#build model in graph
	with tf.Graph().as_default():
		images_placeholder, labels_placeholder = placeholder_inputs(
			FLAGS.batch_size)

		#build graph, computes predictions from model
		logits = mnist.inference(images_placeholder,
			FLAGS.hidden1,
			FLAGS.hidden2)

		#graph the ops for calculation
		##train op, eval correction
		loss = mnist.loss(logits, labels_placeholder)
		train_op = mnist.training(loss, FLAGS.learning_rate)
		eval_correct = mnist.evaluation(logits, labels_placeholder)

		#build summary based on tf
		##add init
		###checkpoint saver
		summary_op = tf.merge_all_summaries()
		init = tf.initialize_all_variables()
		saver = tf.train.Saver()

		#start sess, summary writer, init variables
		sess = tf.Session()
		summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)
		sess.run(init)

		#training loop
		for step in xrange(FLAGS.max_steps):
			start_time = time.time()

		#feed_dict gets actual images and labels
		feed_dict = fill_feed_dict(
			data_sets.train,
			images_placeholder,
			labels_placeholder)


		_, loss_value = sess.run([train_op, loss],
			feed_dict=feed_dict)

		duration = time.time() - start_time

		#write summaries, print status and time
		if step % 100 == 0:
			print('step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))

			#update events file
			summary_str = sess.run(summary_op, feed_dict=feed_dict)
			summary_writer.add_summary(summary_str, step)
			summary_writer.flush()

		#check point saving and evaluation
		if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
			saver.save(sess, FLAGS.train_dir, global_step=step)

        # eval against the training set
        print('Training Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.train)

        print('Validation Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.validation)

        print('Test Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.test)

def main(_):
	run_training()

if __name__ == '_main_':
	tf.app.run()









