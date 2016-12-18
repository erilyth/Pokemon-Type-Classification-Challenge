#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 vishalapr <vishalapr@vishal-Lenovo-G50-70>
#
# Distributed under terms of the MIT license.

"""

"""

from preprocess import *
import tfnet
import warnings
import numpy as np
from sklearn import decomposition
from sklearn.svm import SVC
from random import sample
# Ignore deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

data, labels_temp = get_data('Pokemon.csv')
labels = []
for label in labels_temp:
	labels.append(classes[label])
data = normalize_data(data)

# Split data into 1/10 test and 9/10 train
start_idx = 0
indices = sorted(sample(range(len(data)), len(data)/10))
test_data = []
test_labels = []
train_data = []
train_labels = []

print indices

for i in range(len(data)):
	if start_idx < len(indices) and indices[start_idx] == i:
		test_data.append(data[i])
		test_labels.append(labels[i])
		start_idx += 1
	else:
		train_data.append(data[i])
		train_labels.append(labels[i])

print len(train_data), len(test_data)
tfnet.main(train_data, train_labels, test_data, test_labels)