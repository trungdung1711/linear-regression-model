import numpy as np
from enum import Enum
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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
        self.loss_history = []

    def fit(self, X, y):
        # m training samples, n features
        # convert the raw array into NDArray
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        # (m, 1)
        y = np.asarray(y).reshape(-1, 1)

        # scale for X
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        X = (X - self.X_mean) / (self.X_std + 1e-8)

        self.m, self.n = X.shape
        # initial sampling
        # the weights of the model
        self.w = np.zeros((self.n, 1))
        # the bias of the model
        self.b = 0.0

        # partial gradients
        # matrix
        def dJ_over_dw(error):
            return (1 / self.m) * (X.T @ error)

        def dJ_over_db(error):
            return (1 / self.m) * np.sum(error)

        # loss function (MSE)
        # matrix because of the number of data points
        def J(error):
            return np.sum(error**2) / (2 * self.m)

        previous_loss = float("inf")

        # the traning process using gradient descent
        # to find the parameters that make the J min
        for i in range(self.epochs):
            current_error = X @ self.w + self.b * np.ones((self.m, 1)) - y
            # forward pass
            j = J(current_error)
            # print("-Loss function: ", j)
            self.loss_history.append(j)

            # backward pass
            rate_of_change_J_w = dJ_over_dw(current_error)
            rate_of_change_J_b = dJ_over_db(current_error)

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

        # scale X
        X = (X - self.X_mean) / (self.X_std + 1e-8)

        return X @ self.w + self.b * np.ones((a, 1))

    def parameters(self):
        print("W: ", self.w)
        print("b: ", self.b)

    def view_loss(self):
        df = pd.DataFrame(
            data={"Epoch": range(len(self.loss_history)), "Loss": self.loss_history}
        )

        plt.figure(figsize=(7, 5))
        sns.lineplot(data=df, x="Epoch", y="Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss (J)")
        plt.title("Loss vs Epoch")
        plt.grid(True)

        plt.show()

    def score(self, X, y):
        y = np.asarray(y).reshape(-1, 1)
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        return r2

    def info(self):
        print("- Training samples: ", self.m)
        print("- Number of features: ", self.n)

    def get_params(self, deep=True):
        return {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "method": self.method,
            "tol": self.tol,
        }

    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
