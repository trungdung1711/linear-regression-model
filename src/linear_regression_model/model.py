import numpy as np
from enum import Enum


class LinearRegression:
    class Method(Enum):
        GRADIENT_DESCENT = 1

    def __init__(
        self,
        learning_rate=1e-5,
        epochs=2000,
        method=Method.GRADIENT_DESCENT,
        tol=1e-7,
    ):
        self.learning_rate = learning_rate
        self.m = None
        self.n = None
        self.epochs = epochs
        self.method = method
        self.tol = tol

    def fit(self, X, y):
        # m training samples, n features
        # convert the raw array into NDArray
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        # (m, 1)
        y = np.asarray(y).reshape(-1, 1)

        self.m, self.n = X.shape
        # initial sampling
        # the weights of the model
        self.w = np.zeros((self.n, 1))
        # the bias of the model
        self.b = 0.0

        # partial gradients
        # matrix
        def dJ_over_dw(w, b):
            error = X @ w + b - y
            return (1 / self.m) * (X.T @ error)

        def dJ_over_db(w, b):
            error = X @ w + b - y
            return (1 / self.m) * np.sum(error)

        # loss function (MSE)
        # matrix because of the number of data points
        def J(w, b):
            error = X @ w + b * np.ones((self.m, 1)) - y
            return float((error.T @ error) / (2 * self.m))

        # the traning process using gradient descent
        # to find the parameters that make the J min

        previous_loss = float("inf")

        for i in range(self.epochs):
            # forward pass
            j = J(w=self.w, b=self.b)
            print("-Loss function: ", j)

            # backward pass
            rate_of_change_J_w = dJ_over_dw(self.w, self.b)
            rate_of_change_J_b = dJ_over_db(self.w, self.b)

            # update parameters
            self.w = self.w - self.learning_rate * rate_of_change_J_w
            self.b = self.b - self.learning_rate * rate_of_change_J_b

            # convergence method
            if abs(j - previous_loss) < self.tol:
                print(f"Convergence after {i} epochs")
                break

            previous_loss = j

    def predict(self, X):
        # (a, n) a new samples
        # (n, 1)
        # (a, 1)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        a, _ = X.shape
        return X @ self.w + self.b * np.ones((a, 1))

    def parameters(self):
        print("W: ", self.w)
        print("b: ", self.b)

    def score(self, X, y):
        None

    # def J(self, w, b):
    #     a = (self.X @ w + b * np.ones((self.m, 1)) - y).T
    #     b = self.X @ w + b * np.ones((self.m, 1)) - y
    #     return (1 / (2 * self.m)) * (a @ b)

    def info(self):
        print("- Training samples: ", self.m)
        print("- Number of features: ", self.n)
