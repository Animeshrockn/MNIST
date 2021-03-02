# MNIST
Program for Neural Network from scratch 



Introduction
MNIST ("Modified National Institute of Standards and Technology") is the de facto “Hello World” dataset of computer vision. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification algorithms. As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike. In this program, we aim to correctly identify digits from a dataset of tens of thousands of handwritten images.

MNIST DATASET:
Grayscale images of size 28×28, 
60,000 training images, 10,000 test images, 
Available in various formats, 
We will use the CSV format from http://pjreddie.com/projects/mnist-in-csv/, 
Image format available from http://yann.lecun.com/exdb/mnist/.

Approach
For this Program, we will be using Python as the main programming language to create an Artificial neural network from scratch for predicting, as accurately as we can, digits from handwritten images. In particular, we will be creating a class for neural network (threelayernetwork) along with four methods (__int__, set_weights, train, query and backquery) and call the class and intantiate the objects to create a three layer neural network with below architecture:

NETWORK ARCHITECTURE:
Input layer: 28 × 28 = 784 nodes, 
Hidden layer: 200 nodes (design choice), 
Output layer: 10 nodes (one for each digit), 
The output node with the highest value is taken as the predicted digit.


We will be experimenting with Stochastic Gradient Descent optimizer. However, there are many other optimizers available, but for our program we will use gradient descent only. One importaint parameter in neural network in number of epochs, which we will take as 5 so it wont take much time to run the code. 

In addition, the choice of hidden layer units are completely arbitrary and may not be optimal. This is yet another parameter which we will not attempt to tinker with.Training take 10 minutes approx on a simple laptop which is not bad for a artificial neural network network (“multilayer perceptron”) with one hidden layer!

We will be using query and backquery in our code for verification purpose thought its more of learning and understanding advantage.

BACK QUERY
set a label vector at the output, 
Run the network backwards to get the input, 
Show the result as an image, 
Similar to back-propagating the error, but have to invert sigmoid, 
𝐡(5,') = 𝐖(%)𝜎,' 𝐡(%).
The inverse of the sigmoid is the logit function, 
𝜎,' 𝑦 = ln
𝑦
1 − 𝑦. 
Since the sigmoid is bounded to (0, 1), the input to logit (𝑦) must always be scaled back to this interval.


Result
Following our simulations on the cross validation dataset, it appears that a 3-layered neural network, using stocastic gradient descent will have a accuracy of slightly more than 97%.

Accuracy: >97%

