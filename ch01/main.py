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