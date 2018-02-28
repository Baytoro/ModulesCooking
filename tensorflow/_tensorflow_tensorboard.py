import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 

def add_layer(inputs,in_size,out_size,n_layer,activation_function = None):
	layer_name = "Layer%s" % n_layer
	with tf.name_scope(layer_name):
		with tf.name_scope('Weights'):
			Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
			tf.summary.histogram(layer_name+'/weights',Weights)
		with tf.name_scope('biases'):
			biases = tf.Variable(tf.zeros([1,out_size]) + 0.1,name='b')
			tf.summary.histogram(layer_name+'/biases',biases)
		with tf.name_scope('Wx_plus_b'):
			Wx_plus_b = tf.matmul(inputs,Weights) + biases
	
	if activation_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activation_function(Wx_plus_b)
	tf.summary.histogram(layer_name+'/outputs',outputs)
	return outputs

# data assign
x_data = np.linspace(-1,1,300) [:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('inputs'):
	xs = tf.placeholder(tf.float32,[None,1],name='x_input')
	ys = tf.placeholder(tf.float32,[None,1],name='y_input')

# net connection
l1 = add_layer(xs, 1, 10, 1, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, 2, activation_function=None)

# train
with tf.name_scope('Loss'):
	loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
	tf.summary.scalar('Loss_result',loss)
with tf.name_scope('train'):
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


# important step
sess = tf.Session()
init = tf.initialize_all_variables()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/",sess.graph) # 必须在定义好sess之后定义


sess.run(init)


# plot figure
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.scatter(x_data,y_data)
# plt.ion()
# plt.show()



for i in range(1000):
	sess.run(train_step, feed_dict={xs:x_data,ys:y_data})
	if i % 50 is 0:
		# to see step improvement
		# print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
		# try:
		# 	ax.lines.remove(lines[0])
		# except Exception as e:
		# 	pass
		# prediction_value = sess.run(prediction,feed_dict={xs:x_data})
		# lines = ax.plot(x_data,prediction_value,'r-',lw=5)
		# plt.pause(0.2)
		result = sess.run(merged,feed_dict={xs:x_data,ys:y_data})
		writer.add_summary(result,i)

