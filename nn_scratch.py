import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# create sample dataset
np.random.seed(0)
X, y = datasets.make_moons(200, noise=0.20)

# training data size
N = len(X)

# input size
nn_input_dim = 2   
nn_output_dim = 2  

# training parameters
learning_rate = 0.01
reg_lambda = 0.01
num_epochs = 10000

# calculate loss
def loss(model, X, y):
    w1, w2, b1, b2 = model['w1'], model['w2'], model['b1'], model['b2']

    # forward pass
    z1 = X.dot(w1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(w2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # cross-entropy loss
    correct_logprobs = -np.log(probs[range(N), y])
    data_loss = np.sum(correct_logprobs)

    # add regularization
    data_loss += reg_lambda / 2 * (np.sum(np.square(w1)) + np.sum(np.square(w2)))
    return 1. / N * data_loss

# predict function
def predict(model, x):
    w1, w2, b1, b2 = model['w1'], model['w2'], model['b1'], model['b2']
    z1 = x.dot(w1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(w2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

# build the model
def model_building(nn_hdim, epochs, learning_rate, reg_lambda):
    # initialize weights and biases
    w1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    w2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    model = {}

    # training loop
    for epoch in range(epochs):
        # forward pass
        z1 = X.dot(w1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(w2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # backpropagation
        delta3 = probs
        delta3[range(N), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(w2.T) * (1 - np.power(a1, 2))
        dW1 = X.T.dot(delta2)
        db1 = np.sum(delta2, axis=0, keepdims=True)

        # add regularization
        dW2 += reg_lambda * w2
        dW1 += reg_lambda * w1

        # gradient descent
        w1 += -learning_rate * dW1
        b1 += -learning_rate * db1
        w2 += -learning_rate * dW2
        b2 += -learning_rate * db2

        # update model
        model = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

        # print loss
        if epoch % 1000 == 0:
            print(f"Loss after iteration {epoch}: {loss(model, X, y)}")

    return model

# plot decision boundary
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = predict(model, np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.show()

# main function
def main():
    print("This is a simple classification example using a Neural Network from scratch.")
    nn_hdim = 3 
    model = model_building(nn_hdim, num_epochs, learning_rate, reg_lambda)
    plot_decision_boundary(model, X, y)

if __name__ == "__main__":
    main()
