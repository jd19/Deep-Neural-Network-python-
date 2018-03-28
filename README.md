# Deep-Neural-Network-python-

We implement a two layer neural network for a binary classifier and a multi layer neural network for a multiclass classifier.

## Two layer neural network for a binary classifier

Dataset: MNIST - digits 2 and 3
Data can be downloaded using ('download mnist.sh'). Train a two layer network (1 hidden layer
dimension=500) for binary classication. Train with the following parameters.
learning rate = 0.1
num iterations = 1000

## Multi-layer neural network for multi-class classier(

Data can be downloaded using ('download mnist.sh'). Train a multi-layer neural network to classify
MNIST. The MNIST dataset has 60,000 images which may be too large for batch gradient descent.
Therefore, train with merely 6000 samples and test with 1000 samples.

Modify the num iterations, learning rate and decay rate to improve training.
The program must be able to create and train a multilayer network based on command line arguments.
To create a network with 1 hidden layer of dimensions 800 Run the program as:
python deepMultiClassNetwork starter.py "[784,800]"
The network will have the dimensions [784,800,10]
784 is the input size of digit images (28pix x 28pix = 784)
10 is the number of digits
To create a network with 2 hidden layers of dimensions 800 and 500 Run the
program as:
python deepMultiClassNetwork starter.py "[784,800,500]"
The network will have the dimensions [784,800,500,10]
784 is the input size of digit images (28pix x 28pix = 784)
10 is the number of digits
