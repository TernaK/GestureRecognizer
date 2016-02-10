"""
This script trains a 3-layer neural network (1024-50-10-2) to be able to recognize two gestures.
It uses SGD + cross entropy with constant learning rate.
"""

import numpy as np
import tensorflow as tf
import pickle as pickle
import matplotlib.pyplot as plt

dataset = pickle.load( open( "dataset.pickle", "rb" ) )
train = dataset['train']
trainLabels = dataset['trainLabels']
test = dataset['test']
testLabels = dataset['trainLabels']

#plt.imshow(np.reshape(train[400], [32,32]), cmap = plt.cm.Greys_r)
#plt.show()

x = tf.placeholder(tf.float32, [None,32*32])
W1 = tf.Variable(tf.truncated_normal([32*32, 50], stddev = 0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[50]))
y1 = tf.nn.relu(tf.matmul(x, W1)+b1)

W2 = tf.Variable(tf.truncated_normal([50, 10], stddev = 0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[10]))
y2 = tf.nn.relu(tf.matmul(y1, W2)+b2)

W = tf.Variable(tf.truncated_normal([10, 2], stddev = 0.1))
b = tf.Variable(tf.constant(0.1, shape=[2]))
y = tf.nn.softmax(tf.matmul(y2, W)+b)

y_ = tf.placeholder(tf.float32, [None, 2])

cross_entropy = -tf.reduce_sum(y_*tf.log( tf.clip_by_value(y, 1e-10, 1.0) ))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()

sess.run(init)

stepSize = 20
steps = 40
acc = []
for j in range(10):
	for i in range(steps):
		start = i*stepSize
		end = start + stepSize
		batch_xs, batch_ys = (train[start:end], trainLabels[start:end])

		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
	pred = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))
	acc.append(sess.run(accuracy, feed_dict={x:test[:200], y_:testLabels[:200]}))
	print("Accuracy: ",acc[-1])
			
#print("Predicted:", sess.run(y, feed_dict={x:np.reshape(train[10:20], [10, 32*32])}))
#print("Actual:",trainLabels[10:20])
#sess.close()

#plt.plot(acc)
#plt.show()
