# coding: UTF-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from single_layer_network import SingleLayerNetwork

np.random.seed(20160612)
tf.set_random_seed(20160612)

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

nn = SingleLayerNetwork(1024)

i = 0
for _ in range(2000):
    i += 1
    batch_xs, batch_ts = mnist.train.next_batch(100)
    nn.sess.run(nn.train_step, feed_dict={nn.x: batch_xs, nn.t:batch_ts})
    if i % 100 == 0:
        summary, loss_val, acc_val = nn.sess.run(
            [nn.summary, nn.loss, nn.accuracy],
            feed_dict={nn.x:mnist.test.images, nn.t: mnist.test.labels})
        print('Step: %d, Loss: %f, Accuracy: %f'
              % (i, loss_val, acc_val))
        nn.writer.add_summary(summary, i)