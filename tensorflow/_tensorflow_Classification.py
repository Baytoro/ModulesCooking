import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# define layer
def add_layer(inputs, in_size, out_size, activation_function=None):
	Weights = tf.Variable(tf.random_normal([in_size, out_size]))
	biases  = tf.Variable(tf.zeros([1, out_size]) + 0.1)
	Wx_plus_b = tf.matmul(inputs, Weights) + biases
	if activation_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activation_function(Wx_plus_b)
	return outputs

# define accuracy
def accuracy(v_xs, v_ys):
	global prediction
	y_pre = sess.run(prediction, feed_dict={xs:v_xs}) # 此处不可以另外用add_layer来计算   
	correct_pred = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
	accuracy_rate = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	result = sess.run(accuracy_rate, feed_dict={xs:v_xs, ys:v_ys})
	return result

# define placeholder
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

# add output layer
# l1 = add_layer(xs, 784, 100, activation_function=tf.nn.tanh)
# prediction = add_layer(l1, 100, 10, activation_function=tf.nn.softmax)
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

# error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# importment set up
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(50)  # 100 -> 50
	sess.run(train_step,feed_dict={xs:batch_xs, ys:batch_ys})
	if i % 50 is 0:
		print(accuracy(mnist.test.images, mnist.test.labels))