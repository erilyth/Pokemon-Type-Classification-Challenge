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

train_data, train_labels_temp = get_data('Pokemon.csv')
train_labels = []
for label in train_labels_temp:
	train_labels.append(classes[label])
train_data = normalize_data(train_data)

# Apply Principal Component Analysis to select the best representative feature combinations
pca = decomposition.PCA(n_components=5)
pca.fit(train_data)
train_data = pca.transform(train_data)

# Apply an SVM to check how it performs, we get around 36% accuracy
accuracy = 0.0
svm = SVC()
svm.fit(train_data, train_labels)
for point in range(len(train_data)):
	if svm.predict(train_data[point]) == train_labels[point]:
		accuracy += 1.0
accuracy = accuracy / len(train_data) * 100.0
print accuracy

print tfnet.main(train_data, train_labels)