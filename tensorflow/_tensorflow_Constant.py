import tensorflow as tf


seed = tf.set_random_seed(2)

a = tf.constant(2.0)
a_1 = tf.zeros([2,3])
b = tf.zeros_like(a, dtype = tf.float32)
c = tf.fill([2,3],8.0)
b_1 = tf.ones_like(c)




# tf.linspace(start, stop, num, name=None) # slightly different from np.linspace
# tf.linspace(10.0, 13.0, 4) ==> [10.0 11.0 12.0 13.0]
# tf.range(start, limit=None, delta=1, dtype=None, name='range')
# # 'start' is 3, 'limit' is 18, 'delta' is 3
# tf.range(start, limit, delta) ==> [3, 6, 9, 12, 15]
# # 'limit' is 5
# tf.range(limit) ==> [0, 1, 2, 3, 4]


d = tf.linspace(10., 13., 4)
d_1 = tf.range(10,13,1)
d_2 = tf.range(5)



# tf.random_shuffle(value, seed=None, name=None)
# tf.random_crop(value, size, seed=None, name=None)
# tf.multinomial(logits, num_samples, seed=None, name=None)
# tf.random_gamma(shape, alpha, beta=None, dtype=tf.float32, seed=None, name=None)
 
e =  tf.random_normal([2,3],seed = seed)
f_0 = tf.log([[10., -1.]])
f = tf.multinomial(f_0, 20)

t = tf.constant([[1,2,3],[2,1,3],[1,1,2],[4,3,1]])

with tf.Session() as sess:
	print(tf.reduce_sum(tf.cast(tf.equal(tf.argmax(t,1),tf.constant([1,0,2,0],dtype=tf.int64)),dtype=tf.float32)).eval())
