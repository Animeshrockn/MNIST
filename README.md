MNIST 
========
## Neural Network from scratch <br>
<br>


**Introduction:**
MNIST ("Modified National Institute of Standards and Technology") is the de facto “Hello World” dataset of computer vision. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification algorithms. As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike. In this program, we aim to correctly identify digits from a dataset of tens of thousands of handwritten images.

**MNIST Dataset:**
Grayscale images of size 28×28, 
60,000 training images, 10,000 test images, 
Available in various formats, 
We will use the [CSV format](http://pjreddie.com/projects/mnist-in-csv/), 
[Image format](http://yann.lecun.com/exdb/mnist/).

**Approach:**
For this Program, we will be using Python as the main programming language to create an Artificial neural network from scratch for predicting, as accurately as we can, digits from handwritten images. In particular, we will be creating a class for neural network (threelayernetwork) along with four methods (__int__, set_weights, train, query and backquery) and call the class and intantiate the objects to create a three layer neural network with below architecture:

**Network Architecture:**
Input layer: 28 × 28 = 784 nodes, 
Hidden layer: 200 nodes (design choice), 
Output layer: 10 nodes (one for each digit), 
The output node with the highest value is taken as the predicted digit.
<br>
We will be experimenting with Stochastic Gradient Descent optimizer. However, there are many other optimizers available, but for our program we will use gradient descent only. One importaint parameter in neural network in number of epochs, which we will take as 5 so it wont take much time to run the code. In addition, the choice of hidden layer units are completely arbitrary and may not be optimal. This is yet another parameter which we will not attempt to tinker with.Training take 10 minutes approx on a simple laptop which is not bad for a artificial neural network network (“multilayer perceptron”) with one hidden layer!

**Backquery:**

We will be using query and backquery in our code for verification purpose thought its more of learning and understanding advantage.<br>

- Back-query:
  - Set a label vector at the output
  - Run the network backwards to get the input
  - Show the result as an image

- Similar to back-propagating the error, but have to invert sigmoid <br><br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <a href="https://www.codecogs.com/eqnedit.php?latex=h^{(l-1)}&space;=&space;W&space;^{(l)}&space;\sigma^{(-1)}&space;(h^{(l)})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h^{(l-1)}&space;=&space;W&space;^{(l)}&space;\sigma^{(-1)}&space;(h^{(l)})" title="h^{(l-1)} = W ^{(l)} \sigma^{(-1)} (h^{(l)})" /></a>


- The inverse of the sigmoid is the logit function <br><br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <a href="https://www.codecogs.com/eqnedit.php?latex=\sigma^{(-1)}(y)&space;=&space;ln&space;(&space;\frac{y}{1-y})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sigma^{(-1)}(y)&space;=&space;ln&space;(&space;\frac{y}{1-y})" title="\sigma^{(-1)}(y) = ln ( \frac{y}{1-y})" /></a>

Since the sigmoid is bounded to (0, 1), the input to logit (𝑦) must always be scaled back to this interval.


**Result:**
Following our simulations on the cross validation dataset, it appears that a 3-layered neural network, using stocastic gradient descent will have a accuracy of slightly more than 97%.

## Accuracy: 
Accuracy for the Model is more than 97%.<br><br>

**Credit:**
It's based on the book [Make Your Own Neural Network](https://www.amazon.com/Make-Your-Own-Neural-Network/dp/1530826608/ref=as_li_ss_tl?ie=UTF8&qid=1489506339&sr=8-1-fkmr1&keywords=create+your+own+neural+network+python&linkCode=sl1&tag=natureofcode-20&linkId=c12539edab4fd9b21c4801d1eae57dfc) by Tariq Rashid [book source code](https://github.com/makeyourownneuralnetwork).

<img width="572" alt="Screenshot 2021-05-08 at 3 08 25 PM" src="https://user-images.githubusercontent.com/77103507/117537425-61b9b500-b01e-11eb-9f92-9dca4206070a.png">
