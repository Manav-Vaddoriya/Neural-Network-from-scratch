from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

# create sample dataset
def create_dataset():
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y

# plot the X,y desicion boundary
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.show()

# use the LogisticRegression model to fit the data
def classifier(X,y):
    model = LogisticRegression()
    model.fit(X,y)
    return model

# include everything in a main function
def main():
    print("This is a simple classification example using Logistic Regression.")
    X, y = create_dataset()
    model = classifier(X, y)
    plot_decision_boundary(model, X, y)

if __name__ == "__main__":
    main()