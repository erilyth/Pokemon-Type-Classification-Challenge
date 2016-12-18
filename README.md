# Pokemon Type Prediction

* Given the stats of a Pokemon as the input, predict which type they belong to (their primary type). An example is, Bulbasaur -> Grass. The task, although it seems rather simple, is not that easy since a lot of the stats are very similar in a lot of different types. Pokemon that have two types make it even harder since their stats would be shared.
* There are 19 types in total, so the network performs a 19 class classification
* Implemented a simple 4 layered neural network (2 hidden layers) with a softmax layer at the end
* Uses adam optimization to perform updation
* Computes a top-5 match in the accuracy (ie. If any one of the top 5 classes match the correct output, we consider it as correct). Also performed a few experiments with top-3 and top-1 matches as well

## Pokemon Type Challenge
Pokemon Type Prediction challenge by @Sirajology on [Youtube](https://youtu.be/9MvbNPQiEE8)

## Dependencies
* Tensorflow
* Numpy

## Usage
Run `python main.py` and it would train the network and then run it on a randomly subsampled test dataset (not included in the training) and print the accuracy.
