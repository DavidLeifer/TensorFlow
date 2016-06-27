#easy basics
import tensorflow as tf

#1 X 2 matrix
matrix1 = tf.constant([[3., 3.]])

#2 X 1 matrix
matrix2 = tf.constant([[2.],[2.]])

#multiply
product = tf.matmul(matrix1, matrix2)

sess = tf.Session()

result = sess.run(product)
print(result)

# Enter an interactive TensorFlow Session.
import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])
x.initializer.run()

# Add an op to subtract 'a' from 'x'.  Run it and print the result
sub = tf.sub(x, a)
print(sub.eval())
sess.close()

##variables##
state = tf.Variable(0, name="counter")

#create an op to add one to state
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init_op)
	print(sess.run(state))
	for i in range(3):
		sess.run(update)
		print(sess.run(state))
#do a dance

##fetches##
input1 = tf.constant([3.0])
input2 = tf.constant([2.0])
input3 = tf.constant([5.0])
intermed = tf.add(input2, input3)
mul = tf.mul(input1, intermed)

with tf.Session() as sess:
	result = sess.run([mul, intermed])
	print(result)
#heel, deeg

#feed the beast
input4 = tf.placeholder(tf.float32)
input5 = tf.placeholder(tf.float32)
output = tf.mul(input4, input5)

with tf.Session() as sess:
	print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))
#roll over

