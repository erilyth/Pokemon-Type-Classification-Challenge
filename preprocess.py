#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 vishalapr <vishalapr@vishal-Lenovo-G50-70>
#
# Distributed under terms of the MIT license.

"""

"""

import numpy as np
import math
import random

# Give a class ID to each output type
classes = {}

features = 7
current_class_id = 1

def get_data(data_file):
	global current_class_id
	data = []
	labels = []
	inp_file = open(data_file,'r')
	lines = inp_file.readlines()
	# Ignore the first line since it is just the names of each attribute
	for line in lines[1:]:
		data_point = []
		data_cur = line.split(',')
		# Labels 4 to 10 give us the attributes which we can use to classify
		for i in range(4,len(data_cur)-2):
			data_point.append(data_cur[i])
		# Add the type of the pokemon as a label
		labels.append(data_cur[2])
		data.append(data_point)
		if data_cur[2] not in classes:
			classes[data_cur[2]] = current_class_id
			current_class_id += 1
	return data, labels

def normalize_data(data):
	data = np.asarray(data,dtype=np.float32)
	mean = data.mean(axis=0)
	std = data.std(axis=0)
	data = (data - mean) / std
	return data