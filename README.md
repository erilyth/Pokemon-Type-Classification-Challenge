# Pokemon Type Prediction

* Given the stats of a Pokemon as the input, predict which type they belong to (their primary type). An example is, Bulbasaur -> Grass. The task, although it seems rather simple, is not that easy since a lot of the stats are very similar in a lot of different types. Pokemon that have two types make it even harder since their stats would be shared.
* There are 19 types in total, so the network performs a 19 class classification
* Implemented a simple 4 layered neural network (2 hidden layers) with a softmax layer at the end
* Uses adam optimization to perform updation
* Computes a top-5 match in the accuracy (ie. If any one of the top 5 classes match the correct output, we consider it as correct). Also performed a few experiments with top-3 and top-1 matches as well

## Pokemon Type Challenge
Pokemon Type Prediction challenge by @Sirajology on [Youtube](https://www.youtube.com/watch?v=0xVqLJe9_CY)

## Dependencies
* Tensorflow
* Numpy

## Usage
Run `python main.py` and it would train the network and then run it on a randomly subsampled test dataset (not included in the training) and print the accuracy.

## Results
|ID      |Top-K  |Network-Shape  |Iterations     |Accuracy.avg	|   
|--------|-------|---------------|---------------|--------------|
|1       |5      |(7,128,256,19) |100            |58.7499976158	|
|2       |5      |(7,128,256,19) |500            |60.2500001921	|
|3       |5      |(7,512,256,19) |100            |63.7499988079	|
|4       |5      |(7,512,256,19) |500            |65.9999976158	|
|5       |3      |(7,128,256,19) |100            |41.2499999049	|
|7       |3      |(7,512,256,19) |100            |43.7500000011	|
