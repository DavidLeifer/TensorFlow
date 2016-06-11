####MNIST For ML Experts
####https://www.tensorflow.org/versions/r0.9/tutorials/mnist/pros/index.html
####Machine learning and TensorFlow
####David Leifer 20160610

#Download data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#start a session
import tensorflow as tf
sess = tf.InteractiveSession()

#create placeholder nodes
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#variable model parameters
##784 inpouts, 10 outputs
###b is a 10 dimensional vector
####initialize variables
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
sess.run(tf.initialize_all_variables())

#implement model: multiply vectorized imgs x by weighted
## matrix w add bias b
y = tf.nn.softmax(tf.matmul(x,W) + b)
#cross-entropy (comparing distrubtion) between
##the target and model's prediction
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#Training
##steepest gradient descent- .5 step length
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#each iteration will load 50 training examples
for i in range(1000):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

#Evaluate the Model
#list of booleans
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#which fraction is correct
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#print .9092 percent accuracy
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

# Close the Session when we're done.
sess.close()