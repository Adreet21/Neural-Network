# Neural Network

Given some handwritten numbers, we will try to classify each number correctly. I used Neural Networks to solve this; additionally, I made three different Neural Networks to see which one was more accurate.

## What is a Neural Network?
- A Neural Network consists of neuron* (Nodes) layers and is typically made of three main layers: Input, Hidden, and Output layer.

- Each neuron is its own Linear Regression Model. The weight and biases of the connections between the neurons determine how much influence each input has on the output.

- Data is passed from one layer to the next, which is called the Feed-Forward Network.

- Neural Network uses training data to learn and improve accuracy, using Supervised Learning. As we train the model we evaluate its accuracy using a Cost Function.<br>
This process is known as Gradient Decent.

## *What is a Neuron?
- A Neuron uses a function to give a predicted outcome (Linear Regression Model)<br>
- ŷ = (x₁ * w₁) + (x₂ * w₂) + ... + (xₙ * wₙ) - n<br>
where:<br>
x = The factor with possible values of 0 or 1.<br>
w = The weight of that factor with possible values from 0 to 1.<br>
n = The number of factors used.<br>

## Types of Neural Networks tested
1. MLP (Multilayer Perceptron)
2. CNN (Convolution Neural Network)

MLP1:
A densely-connected feedforward network with a single hidden layer of 512
ReLU units. The output layer is 10 fully connected log softmax
units.

MLP2:
A densely-connected feedforward network with two hidden layers of 128
and 64 ReLU units respectively. The output layer is 10 fully connected log softmax
units.

CNN:
A convolutional neural network with the following layers:
* A layer of 32 convolutional units with a kernel size of 5x5 and a stride of 1,1, with ReLU activation
* A max-pooling layer with a pool size of 2x2 and a stride of 2,2.
* A layer of 64 convolutional units with a kernel size of 5x5 and the default stride, with ReLU activation.
* A max-pooling layer with a pool size of 2x2 and the default stride.
* A Flatten layer (to reshape the image from a 2D matrix into a single long vector)
* A layer of 512 fully-connected linear units with ReLU activation
* A layer of 10 fully-connected linear units with log-softmax activation (the output layer)
* Output layer is 10 fully-connected log-softmax units

## Installation

Make a clone of all the files in the repository.

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install following:

```bash
pip3 install --user torch torchvision 
```
Make sure you are using the correct version of Python (ideally Python 3).<br>
Keep all the documents in the same folder; if not, redirect them accordingly.<br>
Run the cnn.py file

## Output

After successfully running the code in the terminal, there should be many lines of output.<br>
Something similar to this "epoch 1/5 batch   1/938 loss = 2.3016409873962402".
This is just the model training itself and gives the cost function as it goes.
Finally you will see a line like this "*** MLP1 accuracies: test 1 = 0.978 test 2 = 0.5589"
This is the test result of the model after it has been trained.

It will repeat the same thing for MLP2 and CNN, with its test results accordingly.

After that, we can compare the accuracy of each of the models to see which one performed better.

## Not functioning?
If you run into difficulties or errors in the code please feel free to reach out.<br>
Email: contact@shahmeer.xyz
