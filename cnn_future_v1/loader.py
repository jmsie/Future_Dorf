import tensorflow as tf
import numpy as np
import os
import re
import requests
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed

class DataSet:
	def __init__(self, images, labels, dtype=dtypes.float32):
		self._images = images
		self._labels = labels
		self._epochs_completed = 0
		self._index_in_epoch = 0
		self._total_batches = images.shape[0]

	@property
	def images(self):
		return self._images

	@property
	def labels(self):
		return self._labels

	@property
	def total_batches(self):
		return self._total_batches

	@property
	def epochs_completed(self):
		return self._epochs_completed

	def next_batch(self, batch_size, shuffle=True):
		start = self._index_in_epoch

		# first epoch shuffle
		if self._epochs_completed==0 and start==0 and shuffle:
			perm0 = np.arange(self._total_batches)
			np.random.shuffle(perm0)
			self._images = self.images[perm0]
			self._labels = self.labels[perm0]

		# next epoch
		if start+batch_size <= self._total_batches:
			self._index_in_epoch += batch_size
			end = self._index_in_epoch
			return self._images[start:end], self._labels[start:end]

		# if the epoch is ending
		else:
			self._epochs_completed += 1
			# store what is left of this epoch
			batches_left = self._total_batches - start
			images_left = self._images[start:self._total_batches]
			labels_left = self._labels[start:self._total_batches]

			# shuffle for new epoch
			if shuffle:
				perm = np.arange(self._total_batches)
				np.random.shuffle(perm)
				self._images = self._images[perm]
				self._labels = self._labels[perm]

			# start next epoch
			start = 0
			self._index_in_epoch = batch_size - batches_left
			end = self._index_in_epoch
			images_new = self._images[start:end]
			labels_new = self._labels[start:end]
			return np.concatenate((images_left, images_new), axis=0), np.concatenate((labels_left, labels_new), axis=0)

class Base():
	def __init__(self, train, test):
		self._train = train
		self._test = test

	@property
	def train(self):
		return self._train

	@property
	def test(self):
		return self._test

# future data loading
def load_future_data(path, moving_window=128, columns=5, train_test_ratio=4.0):
	# parse data from path
	def parse_data(line_num=1000):
		_input = requests.get(path, stream=True)
		line_count = 0
		future_set = []
		for line in _input.iter_lines():
			if line_count==line_num:
				break
			if line and line_count!=0:
				decoded_line = line.decode('utf-8')
				future_set.append(re.split(',', decoded_line)[2:])
			line_count += 1
		future_set = np.array(future_set, dtype=np.float32)
		return future_set

	# process a single file's data into usable arrays
	def process_data(data):
		future_set = np.zeros([0, moving_window, columns])
		label_set = np.zeros([0, 2])
		for idx in range(data.shape[0]-(moving_window+5)):
			future_set = np.concatenate((future_set, np.expand_dims(data[range(idx, idx+moving_window), :], axis=0)), axis=0)
			if data[idx+(moving_window+5),3] > data[idx+(moving_window),3]:
				lbl = [[1.0, 0.0]]
			else:
				lbl = [[0.0, 1.0]]
			label_set = np.concatenate((label_set, lbl), axis=0)

		return future_set, label_set

	# parse data from path
	futures_set = np.zeros([0, moving_window, columns])
	labels_set = np.zeros([0, 2])
	futures_set, labels_set = process_data(parse_data())

	# shuffling the data
	idx = np.arange(labels_set.shape[0])
	np.random.shuffle(idx)
	futures_set = futures_set[idx]
	labels_set = labels_set[idx]

	# normalize the data
	futures_set_ = np.zeros(futures_set.shape)
	for i in range(futures_set.shape[0]):
		min = np.min(futures_set[i], axis=0)
		max = np.max(futures_set[i], axis=0)
		futures_set_[i] = (futures_set[i]-min)/(max-min)
	futures_set = futures_set_

	# selecting 1/(train_test_ratio+1) for testing, and train_test_ratio/(train_test_ratio+1) for training
	train_test_idx = int((1.0/(train_test_ratio+1.0))*labels_set.shape[0])
	train_futures = futures_set[train_test_idx:]
	train_labels = labels_set[train_test_idx:]
	test_futures = futures_set[:train_test_idx]
	test_labels = labels_set[:train_test_idx]

	train = DataSet(train_futures, train_labels)
	test = DataSet(test_futures, test_labels)
	base = Base(train=train, test=test)
	return base

# if __name__ == '__main__':
# 	db = load_future_data('http://dorfcapital.com/toolbox/data/EXF1-Minute-Trade.txt')
# 	images, labels = db.train.next_batch(10)
# 	print(images.shape, labels.shape)
# 	print(images, labels)