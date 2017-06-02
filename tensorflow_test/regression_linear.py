import tensorflow as tf
import numpy as np

# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
x1_data = np.random.rand(100).astype(np.float32)
x2_data = np.random.rand(100).astype(np.float32)
M = 100
N = 2
w_data = np.mat([[1.0, 3.0]]).T
b_data = 10
# x_data = np.random.randn(M, N).astype(np.float32)
x_data=np.concatenate((x1_data,x2_data )).reshape(100,2)
y_data = np.mat(x_data) * w_data + 10 + np.random.randn(M, 1) * 0.33
# y_data = x1_data+x2_data*0.3+ 0.5
# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but TensorFlow will
# figure that out for us.)
x = tf.placeholder(tf.float32, [None,N])
W = tf.Variable(tf.zeros([N, 1]))
b = tf.Variable(tf.zeros([1]))
y = tf.matmul(x, W) + b
# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None,1])
# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
# Before starting, initialize the variables.  We will 'run' this first.
init = tf.global_variables_initializer()
# Launch the graph.
sess = tf.Session()
sess.run(init)
# Fit the line.
for step in range(201):
    sess.run(train, feed_dict={x:x_data , y_: y_data})
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))


# Learns best fit is W: [0.1], b: [0.3]