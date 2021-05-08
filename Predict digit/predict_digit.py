# Modified from https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork
# Code for "Make Your Own Neural Network" by T. Rashid

# python notebook for Make Your Own Neural Network
# code for a 3-layer neural network, and code for learning the MNIST dataset
# (c) Tariq Rashid, 2016
# license is GPLv2

import numpy as np
# scipy.special for the sigmoid function expit()
import scipy.special
import scipy.misc
# argument parsing
import argparse
# OpenCV
import cv2


# ---------------- threeLayerNetwork class ----------------
# neural network class definition
class threeLayerNetwork:
    
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.n1 = inputnodes
        self.n2 = hiddennodes
        self.n3 = outputnodes
        
        # link weight matrices, W2 and W3
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w12
        # w21 w22 etc
        # initialize weights to zero-mean normal with std = 1/sqrt(n) 
        self.W2 = np.random.normal(0.0, pow(self.n1, -0.5), (self.n1, self.n2))
        self.W3 = np.random.normal(0.0, pow(self.n2, -0.5), (self.n2, self.n3))

        # learning rate
        self.lr = learningrate
        
        # activation function is the sigmoid function (called expit in scipy)
        self.activation_function = lambda x: scipy.special.expit(x)
        
        pass

    
    # set weights to given numpy arrays
    def set_weights(self, W2, W3):
        self.W2 = W2
        self.W3 = W3

        pass


    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        h1 = np.array(inputs_list, ndmin=2).T
        y = np.array(targets_list, ndmin=2).T

        # calculate outputs of the hidden layer
        h2 = self.activation_function(np.dot(self.W2.T, h1))

        # calculate outputs of the output layer
        h3 = self.activation_function(np.dot(self.W3.T, h2))

        # output errors
        e3 = y - h3
        # output deltas
        delta3 = -2 * e3 * h3 * (1.0 - h3)
        # hidden deltas (before updating output weights)
        delta2 = np.dot(self.W3, delta3)  * h2 * (1.0 - h2)

        # update the weights for the links between the hidden and output layers
        self.W3 -= self.lr * np.outer(h2, delta3)
        
        # update the weights for the links between the input and hidden layers
        self.W2 -= self.lr * np.outer(h1, delta2)
        
        pass

    
    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        h1 = np.array(inputs_list, ndmin=2).T
        
        # calculate outputs of the hidden layer
        h2 = self.activation_function(np.dot(self.W2.T, h1))
        
        # calculate outputs of the output layer
        h3 = self.activation_function(np.dot(self.W3.T, h2))
        
        return h3


    # backquery the neural network
    # we'll use the same termnimology to each item, 
    # eg target are the values at the right of the network, albeit used as input
    # eg hidden_output is the signal to the right of the middle nodes
    def backquery(self, targets_list):
        # transpose the targets list to a vertical array
        h3 = numpy.array(targets_list, ndmin=2).T
        
        # calculate the signal at the output of the hidden layer
        h2 = numpy.dot(self.W3, self.inverse_activation_function(h3))
        # scale them back to 0.01 to .99
        h2 -= numpy.min(h2)
        h2 /= numpy.max(h2)
        h2 *= 0.98
        h2 += 0.01
        
        # calculate the signal into the hidden layer
        hidden_inputs = self.inverse_activation_function(h2)
        
        # calculate the signal at the input layer
        h1 = numpy.dot(self.W2, self.inverse_activation_function(h2))
        # scale them back to 0.01 to .99
        h1 -= numpy.min(h1)
        h1 /= numpy.max(h1)
        h1 *= 0.98
        h1 += 0.01
        
        return h1


# ---------------- end of threeLayerNetwork class ----------------

# parse input argument(s)
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True)
args = vars(ap.parse_args())

# get image filename
image_filename = args["image"]

# load image data from png files into an array
#img_array = scipy.misc.imread(image_filename, flatten = True)
# load grayscale image using OpenCV
img_array = cv2.imread(image_filename,0)

# reshape from 28x28 to list of 784 values, invert values
img_data  = 255.0 - img_array.reshape(784)
    
# then scale data to range from 0.01 to 1.0
inputs = (img_data / 255.0 * 0.99) + 0.01


# create the network
# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# learning rate  
learning_rate = 0.1

# create the network and load weights
n = threeLayerNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)
W2_pretrained = np.load('W2.npy')
W3_pretrained = np.load('W3.npy')
n.set_weights(W2_pretrained, W3_pretrained)


# apply inputs to the neural network to make a prediction
outputs = n.query(inputs)
prediction = np.argmax(outputs)
print(prediction)