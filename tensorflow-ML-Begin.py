####MNIST For ML Beginners
####https://www.tensorflow.org/versions/r0.9/tutorials/mnist/beginners/index.html
####Machine learning and TensorFlow
####David Leifer 20160610

import tensorflow as tf

##Implementing the Regression
##describe interacting operations by manipulating symbolic variables
#x is a placeholder, inputing imgs into a 784-d vector
x = tf.placeholder(tf.float32, [None, 784])

#create variables- W and b are tensors full of zeros
#we are going to learn W and b
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#implement model, supply softmax
y = tf.nn.softmax(tf.matmul(x, W) + b)

##Training
#add a new placeholder
y_ = tf.placeholder(tf.float32, [None, 10])
#implement cross-entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#apply choice of alg: https://www.tensorflow.org/versions/r0.9/api_docs/python/train.html#optimizers
##to modify the variables/reduce cost
###.5 learning rate
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init = tf.initialize_all_variables()

#launch model in a session
sess = tf.Session()
sess.run(init)

#Start Training
#train_step fedding batches data to replace the placeholders
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#Evaluating the Model
#Generate list of booleans
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#Determine which fraction are correct
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

##0.9204