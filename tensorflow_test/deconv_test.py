import tensorflow as tf
tf.set_random_seed(1)
x = tf.random_normal(shape=[1,3,3,1])
#正向卷积的kernel的模样
kernel = tf.random_normal(shape=[2,2,3,1])

# strides 和padding也是假想中 正向卷积的模样。当然，x是正向卷积后的模样
y = tf.nn.conv2d_transpose(x,kernel,output_shape=[1,5,5,3],
    strides=[1,2,2,1],padding="SAME")
# 在这里，output_shape=[1,6,6,3]也可以，考虑正向过程，[1,6,6,3]
# 通过kernel_shape:[2,2,3,1],strides:[1,2,2,1]也可以
# 获得x_shape:[1,3,3,1]
sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

print(y.eval(session=sess))
