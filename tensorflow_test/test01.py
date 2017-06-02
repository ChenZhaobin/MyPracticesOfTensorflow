# right
import tensorflow as tf

w1 = tf.Variable([[1,2]])
w2 = tf.Variable([[3,4]])
res = tf.matmul(w1, [[2],[1]])
grads = tf.gradients(res,[w1])

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(res))
    re = sess.run(grads)
    print(re)