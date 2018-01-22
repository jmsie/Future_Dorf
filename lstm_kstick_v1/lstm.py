import os
import sys
import datetime
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import warnings
import random
from sklearn import mixture
from fetch import Future
from classifier import bezier, calculate_slope, merge_kstick, GMMClassifier

class LSTM:
	def __init__(self, input_dims, output_dims, lr=0.001, seq_size=20, batch_sz=200):
		self.seq_size = seq_size
		self.input_dims = input_dims
		self.output_dims = output_dims
		self.hidden_layer_one_size = 64
		self.hidden_layer_two_size = 128
		self.learning_rate = lr
		self.batch_sz = batch_sz
		self.input_one_hot = np.zeros((self.batch_sz, self.seq_size, self.input_dims))
		self.output_one_hot = np.zeros((self.batch_sz, self.output_dims))

	def _create_lstm_cell(self):
		W_1 = tf.Variable(tf.random_normal([self.hidden_layer_one_size, self.hidden_layer_two_size], name="W_1"))
		b_1 = tf.Variable(tf.random_normal([self.hidden_layer_two_size], name="b_1"))
		W_2 = tf.Variable(tf.random_normal([self.hidden_layer_two_size, self.output_dims], name="W_2"))
		b_2 = tf.Variable(tf.random_normal([self.output_dims], name="b_2"))

		with tf.variable_scope("cell"):
			lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_layer_one_size)
			mlstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*1)

		with tf.variable_scope("rnn"):
			outputs, states = tf.nn.dynamic_rnn(mlstm_cell, self.X, dtype=tf.float32)

		h_state = outputs[:,self.seq_size-1,:]
		out_layer1 = tf.matmul(h_state, W_1)+b_1
		out_layer2 = tf.nn.softmax(tf.matmul(out_layer1, W_2) + b_2)

		return out_layer2

	def _create_variable(self):
		with tf.name_scope(name="data"):
			self.X = tf.placeholder(tf.float32, [None, self.seq_size, self.input_dims], name="X")
			self.Y = tf.placeholder(tf.float32, [None, self.output_dims], name="Y")	

	def fit(self, X, Y, iteration=100, print_period=1):
		self._create_variable()
		self.pre_y = self._create_lstm_cell()
		self.cost = -tf.reduce_mean(self.Y*tf.log(self.pre_y))
		self.train_optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
		
		costs = []
		current_idx = 0
		with tf.Session() as sess:
			sess.run(tf.initialize_all_variables())
			while current_idx<len(X):
				next_idx = min(current_idx+self.batch_sz, len(X))
				self._input_one_hot(X[current_idx:next_idx], 1)
				self._output_one_hot(Y[current_idx:next_idx], 1)

				train_x = self.input_one_hot[0:next_idx-current_idx]
				train_y = self.output_one_hot[0:next_idx-current_idx]
				predict_y = sess.run(self.pre_y, feed_dict={self.X: train_x})
				accuracy = np.mean(np.argmax(predict_y, axis=1)==np.argmax(train_y, axis=1))
				print("AC: ", accuracy, np.argmax(predict_y, axis=1), np.argmax(train_y, axis=1))
				for step in range(iteration):
					_, c = sess.run([self.train_optim, self.cost], feed_dict={self.X: train_x, self.Y: train_y})
					print(current_idx, step, c)

				# predict_y = sess.run(self.pre_y, feed_dict={self.X: train_x})
				# accuracy = np.mean(np.argmax(predict_y, axis=1)==np.argmax(train_y, axis=1))
				# print("AC: ", accuracy)

				self._input_one_hot(X[current_idx:next_idx], 0)
				self._output_one_hot(Y[current_idx:next_idx], 0)
				current_idx = next_idx

	def _input_one_hot(self, data, value=1):
		for idx1 in range(len(data)):
			for idx2 in range(len(data[0])):
				self.input_one_hot[idx1, idx2, data[idx1][idx2]] = value

	def _output_one_hot(self, data, value=1):
		for idx in range(len(data)):
			self.output_one_hot[idx, data[idx]] = value

if __name__ == "__main__":
    test = Future('http://dorfcapital.com/asset/data/MXF1-Minute-Trade.txt')
    data = test.get_Kstick_data(time_range=5, data_length=15000)
    Kstick = GMMClassifier(data)
    Kstick.fit()

    train_data = []
    train_data_num = 2000
    iteration = 1
    while len(train_data) < train_data_num:
    	train_input = Kstick.transform(data[max(0, iteration-40):iteration], number_data=20)
    	train_target = Kstick.predict(data[iteration])
    	iteration += 1
    	if train_input==False:
    		continue
    	train_data.append([train_input, train_target])
    random.shuffle(train_data)
    train_data = np.array(train_data)

    lstm = LSTM(Kstick.Kstick_input_number, Kstick.Kstick_output_number, seq_size=20)
    lstm.fit(train_data[:,0].tolist(), train_data[:,1])




	