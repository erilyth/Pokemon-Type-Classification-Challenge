#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 vishalapr <vishalapr@vishal-Lenovo-G50-70>
#
# Distributed under terms of the MIT license.

"""

"""

import tensorflow as tf
import random

# Hyperparameters
learning_rate = 0.001
training_epochs = 500

# Network
n_hidden_1 = 512 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 7 # After performing PCA the input size becomes 5
n_classes = 19 # The different types of a Pokemon

def main(train_data, train_labels, test_data, test_labels):
	# Create a one hot representation of the labels
	train_labels_new = []
	test_labels_new = []
	for lab in train_labels:
		label_new = []
		for lab_test in range(1,20):
			if lab == lab_test:
				label_new.append(1)
			else:
				label_new.append(0)
		train_labels_new.append(label_new)
	for lab in test_labels:
		label_new = []
		for lab_test in range(1,20):
			if lab == lab_test:
				label_new.append(1)
			else:
				label_new.append(0)
		test_labels_new.append(label_new)

	X = tf.placeholder("float", [None, n_input])
	Y = tf.placeholder("float", [None, n_classes])
	weights = {
	    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
	    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
	}
	biases = {
	    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
	    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
	    'out': tf.Variable(tf.random_normal([n_classes]))
	}

	h_layer_1 = tf.add(tf.matmul(X, weights['h1']), biases['b1'])
	h_layer_1 = tf.nn.relu(h_layer_1)
	h_layer_2 = tf.add(tf.matmul(h_layer_1, weights['h2']), biases['b2'])
	h_layer_2 = tf.nn.relu(h_layer_2)
	out_layer = tf.matmul(h_layer_2, weights['out']) + biases['out']
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out_layer, Y))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

	# Initialize tensorflow
	init = tf.initialize_all_variables()

	with tf.Session() as ses:
		ses.run(init)
		for ep in range(training_epochs):
			loss_avg = 0
			for point in range(len(train_data)):
				_, c = ses.run([optimizer, cost], feed_dict={X:[train_data[point]], Y:[train_labels_new[point]]})
				loss_avg += c
			print "Epoch " + str(ep) + ", " + "Loss " + str(loss_avg/len(train_data))

		# Test the trained model
		correct_pred2 = tf.nn.in_top_k(out_layer, tf.cast(tf.argmax(Y,1), "int32"), 5)
		accuracy2 = tf.reduce_mean(tf.cast(correct_pred2, "float"))
		print "Accuracy " + str(accuracy2.eval({X:test_data, Y:test_labels_new})*100)
