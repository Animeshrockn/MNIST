#Welcome to my world
# Code for "Make Your Own Neural Network" by T. Rashid

#import numpy
import numpy as np
#scipy.special for the isgmoid function
import scipy.special
# to measure the time
from timeit import default_timer as timer


# ----------------------------three layered network class--------------------
#class for three layered network
class threelayernetwork:

	#initialize the neural network
	def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
		#set number of nodes in each layer
		self.n1 = inputnodes
		self.n2 = hiddennodes
		self.n3 = outputnodes

		# link weight metrices, W2 and W3
		# weight inside the array are w_ij
		# w11,w12,w21,w22
		# initialize random weight
		self.W2 = np.random.normal(0.0, pow(self.n1, -0.5), (self.n1, self.n2))
		self.W3 = np.random.normal(0.0, pow(self.n2, -0.5), (self.n2, self.n3))

		#learning rate
		self.lr = learningrate

		#activation function i.e. sigmoid function
		self.activation_function = lambda x: scipy.special.expit(x)

		pass

	#set weight to array
	def set_weights(self,W2,W3):
		self.W2 = W2
		self.W3 = W3

		pass

	#train the neural networks
	def train(self, inputs_list, targets_list):
		#convert input to 2nd array
		h1 = np.array(inputs_list, ndmin=2).T
		y = np.array(targets_list, ndmin=2).T


		#calculate output of hidden layer
		h2 = self.activation_function(np.dot(self.W2.T, h1))

		#calculate output of output layer
		h3 = self.activation_function(np.dot(self.W3.T, h2))

		#output errors
		e3 = y - h3
		#output deltas
		delta3 = -2 * e3 *h3 * (1.0 - h3)
		#hidden delta
		delta2 = np.dot(self.W3, delta3) * h2 * (1.0 - h2)

		#update the weight for the links b/w hidden and o/p layer
		self.W3 -= self.lr * np.outer(h2, delta3)

		# update the weight for the link b/w input and hidden layer
		self.W2 -= self.lr * np.outer(h1, delta2)

		pass


	def query(self, inputs_list):
		#convert input list to 2D array
		h1 = np.array(inputs_list, ndmin=2).T

		#calculate output of hidden layer
		h2 = self.activation_function(np.dot(self.W2.T, h1))

		#calculate output of output layer
		h3 = self.activation_function(np.dot(self.W3.T, h2))

		return h3


	#backquery the network 

	def backquery(self, targets_list):
		#transpose the targets list to a vertical array
		h3 = numpy.array(targets_list, ndmin=2).T

		#calculate the signal at output of hidden layer
		h2 = numpy.dot(self.W3, self.inverse_activation_function(h3))

		#scale them back to 0.01 and 0.99
		h1 -= numpy.min(h1)
		h1 /= numpy.max(h1)
		h1 *= 0.98
		h1 += 0.01

		return h1


#--------------------------end of three layered Network class---------------------


#start the timer
start_t = timer()

#number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

#learning rate
learning_rate = 0.1

#create instance of neural network
n = threelayernetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

#load the mnist training set CSV to list

training_data_file = open('mnist_csv/mnist_train.csv', 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()


#train the neural network

#epochs
epochs = 5

for e in range(epochs):
	print(f'epoch {e}')
	for record in training_data_list:
		#split the record by the ',' commas
		data_sample = record.split(',')
		#scale and shift the inputs
		inputs = (np.asfarray(data_sample[1:]) / 255.0 * 0.99) + 0.01
		#create the target output values
		targets = np.zeros(output_nodes) + 0.01
		#data sample[0] is the taget set for the record
		targets[int(data_sample[0])] = 0.99
		#run gradient decend based on this data sample
		n.train(inputs, targets)
		pass
	pass

#save the weights
np.save('W2.npy', n.W2)
np.save('W3.npy', n.W3)

#test the neural network

#load the mnist test data CSV file into a list
test_data_file = open("mnist_csv/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

#scorecard for how well network performed
scorecard = []

# go through all the data in the test data set, one by one
for record in test_data_list:
	#split the record by the ',' commas
	data_sample = record.split(',')
	# correct answer is the first value
	correct_label = int(data_sample[0])
	# scale and shift the inputs
	inputs = (np.asfarray(data_sample[1:]) / 255.0 * 0.99) + 0.01
	# query the network 
	outputs = n.query(inputs)
	# the index of the highest value corresponds to the label
	label = np.argmax(outputs)
	#append correct or incorrect to list
	if(label == correct_label):
		#network answer matches correct answer
		scorecard.append(1)
	else:
		scorecard.append(0)
		pass

	pass

#calculate the accuracy
scorecard_array = np.asarray(scorecard)
print(f'accuracy = {(scorecard_array.sum() / scorecard_array.size) * 100}%')

#stop the timer
end_t = timer()
time1 = end_t - start_t
print(f'elapsed time = {time1} seconds')















































