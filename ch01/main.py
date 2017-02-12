# coding: UTF-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = tf.placeholder(tf.float32, [None, 5])
w = tf.Variable(tf.zeros([5, 1]))

y = tf.matmul(x, w)

t = tf.placeholder(tf.float32, [None, 1])

loss = tf.reduce_sum(tf.square(y-t))

train_step = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

train_t = np.array([5.2, 5.7, 8.6, 14.9, 18.2, 20.4, 25.5, 26.4,
                    22.8, 17.5, 11.1, 6.6])
train_t = train_t.reshape([12,1])

train_x = np.zeros([12,5])

for row, month in enumerate(range(1, 13)):
    for col, n in enumerate(range(0, 5)):
        train_x[row][col] = month**n

i = 0
for _ in range(100000):
    i += 1
    sess.run(train_step, feed_dict={x:train_x, t:train_t})
    if i % 10000 == 0:
        loss_val = sess.run(loss, feed_dict={x:train_x, t:train_t})
        print('Step: %d, Loss: %f' % (i, loss_val))