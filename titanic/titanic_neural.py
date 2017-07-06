import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import sklearn as sk
from sklearn import preprocessing

# some functions for quick variable creation
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev = 0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape = shape))

# hyperparameters we will use
learning_rate = 0.001
hidden_layer_neurons = 100
num_iterations = 10000

# placeholder variables
x = tf.placeholder(tf.float32, shape = [None, 8]) # none = the size of that dimension doesn't matter. why is that okay here? 
y_ = tf.placeholder(tf.float32, shape = [None, 1])

# create our weights and biases for our first hidden layer
W_1, b_1 = weight_variable([8, hidden_layer_neurons]), bias_variable([hidden_layer_neurons])

# compute activations of the hidden layer
h_1 = tf.nn.relu(tf.matmul(x, W_1) + b_1)

# create our weights and biases for our output layer
W_2, b_2 = weight_variable([hidden_layer_neurons, 1]), bias_variable([1])
# compute the of the output layer
y = tf.matmul(h_1,W_2) + b_2

# define our loss function as the cross entropy loss
cost_loss = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)) + (1-y_)*tf.log(tf.clip_by_value(1-y,1e-10,1.0)))

# create an optimizer to minimize our cross entropy loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_loss)

# functions that allow us to gauge accuracy of our model
correct_predictions = tf.equal(tf.round(y), y_) # creates a vector where each element is T or F, denoting whether our prediction was right
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32)) # maps the boolean values to 1.0 or 0.0 and calculates the accuracy

# we will need to run this in our session to initialize our weights and biases. 
init = tf.global_variables_initializer()

train = pd.read_csv('clean_train.csv')
train = train.values
this_x = train[:, 2:]
this_y = train[:, 1]
this_y = np.reshape(this_y, (this_y.shape[0],1))

test = pd.read_csv('clean_test.csv')
test = test.values
test_x = test[:, 2:]
print(this_x)

# launch a session to run our graph defined above. 
with tf.Session() as sess:
    sess.run(init) # initializes our variables
    for i in range(num_iterations):
        optimizer.run(feed_dict = {x: this_x, y_: this_y})
        # every 100 iterations, print out the accuracy
        if i % 100 == 0:
            # accuracy and loss are both functions that take (x, y) pairs as input, and run a forward pass through the network to obtain a prediction, and then compares the prediction with the actual y.
            acc = accuracy.eval(feed_dict = {x: this_x, y_: this_y})
            loss = cost_loss.eval(feed_dict = {x: this_x, y_: this_y})
            print("Epoch: {}, accuracy: {}, loss: {}".format(i, acc, loss))

            






