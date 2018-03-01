# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np


# Save to file
# W = tf.Variable([[1,2,3],[3,4,5]], dtype=tf.float32, name='weignts')
# b = tf.Variable([[1,2,3]], dtype=tf.float32, name='bias')

# Restore to Variable
# redifine the same shape and same type
W = tf.Variable(np.arange(6).reshape((2,3)), dtype=tf.float32)
b = tf.Variable(np.arange(3).reshape((1,3)), dtype=tf.float32)

# init = tf.global_variables_initializer() # for Save
# init no need for Restore

saver = tf.train.Saver()

with tf.Session() as sess:
	# sess.run(init)
	# save_path = saver.save(sess,"my_net/save_net.ckpt")
    saver.restore(sess, "my_net/save_net.ckpt")
    print("weignts:\n",sess.run(W))
    print("bias:\n", sess.run(b))