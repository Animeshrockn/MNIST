# MNIST
Program for Neural Network from scratch 

MNIST dataset:
• Grayscale images of size 28×28
• 60,000 training images, 10,000 test images
• Available in various formats
o We will use the CSV format from http://pjreddie.com/projects/mnist-in-csv/
o Image format available from http://yann.lecun.com/exdb/mnist/

Network architecture:
• Input layer: 28 × 28 = 784 nodes
• Hidden layer: 200 nodes (design choice)
• Output layer: 10 nodes (one for each digit)
o The output node with the highest value is taken as the predicted digit

Training takes ~12-13 minutes on a typical laptop
• Accuracy: >97%
• Not bad for a simple network (“multilayer perceptron”) with one hidden layer!

• Back-query:
o set a label vector at the output
o run the network backwards to get the input
o show the result as an image
• Similar to back-propagating the error, but have to invert sigmoid
𝐡(5,') = 𝐖(%)𝜎,' 𝐡(%)
• The inverse of the sigmoid is the logit function
𝜎,' 𝑦 = ln
𝑦
1 − 𝑦
• Since the sigmoid is bounded to (0, 1), the input to logit (𝑦) must always be
scaled back to this interval
