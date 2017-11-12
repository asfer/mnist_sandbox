import numpy as np
import tensorflow as tf

from data import mnist

m = 60000
w0, h0, c0 = 28, 28, 1

X_train = mnist.train_data.astype(np.float32).reshape((m, w0, h0, c0))

# one hot encoding
Y_train = np.eye(10)[mnist.train_labels[:, 0]]
Y_train = Y_train.astype(np.float32)


X = tf.placeholder(tf.float32, shape=(m, w0, h0, c0), name='X')
Y = tf.placeholder(tf.float32, shape=(m, 10), name='Y')

W1 = tf.get_variable("W1", [5, 5, 1, 6],
                     initializer=tf.contrib.layers.xavier_initializer())

W2 = tf.get_variable("W2", [5, 5, 6, 16],
                     initializer=tf.contrib.layers.xavier_initializer())

Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='VALID')
A1 = tf.nn.relu(Z1)
P1 = tf.nn.max_pool(A1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='VALID')
A2 = tf.nn.relu(Z2)
P2 = tf.nn.max_pool(A2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

P2 = tf.contrib.layers.flatten(P2)
Z3 = tf.contrib.layers.fully_connected(P2, num_outputs=120)
Z4 = tf.contrib.layers.fully_connected(Z3, num_outputs=84)
Z5 = tf.contrib.layers.fully_connected(Z4, num_outputs=10, activation_fn=None)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z5, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

num_epochs = 5
with tf.Session() as session:
    init = tf.global_variables_initializer()
    session.run(init)

    for epoch in range(num_epochs):
        _, current_cost = session.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})
        print('%f' % current_cost)
