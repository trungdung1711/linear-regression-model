import numpy as np
from enum import Enum
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class LogisticRegression:
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
        self.N = None
        self.d = None
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

        self.N, self.d = X.shape
        # initial sampling
        # the weights of the model
        self.w = np.zeros((self.d, 1))
        # the bias of the model
        self.b = 0.0

        previous_loss = float("inf")

        def p_hat(X, w, b):
            # return (N, 1)
            # capture the w and b for the model
            return 1 / (1 + np.exp(-(X @ w + b * np.ones((self.N, 1)))))

        def L(y, p_hat):
            return -np.mean(
                y * np.log(p_hat + 1e-15) + (1 - y) * np.log(1 - p_hat + 1e-15)
            )

        # parameters as w, b
        # the loss depends on w and b
        def dL_over_dw(p_hat):
            return (1 / self.N) * (X.T @ (p_hat - y))

        def dL_over_db(p_hat):
            r = p_hat - y
            return np.mean(r)

        # the traning process using gradient descent
        # to find the parameters that make the J min
        for i in range(self.epochs):
            # forward pass
            current_predit = p_hat(X, self.w, self.b)
            Loss = L(y, current_predit)
            print("-Loss function: ", Loss)
            self.loss_history.append(Loss)

            # backward pass
            rate_of_change_J_w = dL_over_dw(current_predit)
            rate_of_change_J_b = dL_over_db(current_predit)

            # update parameters
            self.w = self.w - self.learning_rate * rate_of_change_J_w
            self.b = self.b - self.learning_rate * rate_of_change_J_b

            # convergence method
            if abs(Loss - previous_loss) < self.tol:
                print(f"Convergence after {i} epochs")
                break

            previous_loss = Loss

    def predict(self, X):
        # (a, n) a new samples
        # (n, 1)
        # (a, 1)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        # a samples
        a, _ = X.shape
        prob = 1 / (1 + np.exp(-(X @ self.w + self.b * np.ones((a, 1)))))
        return (prob >= 0.5).astype(int)

    def parameters(self):
        print("W: ", self.w)
        print("b: ", self.b)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def view_loss(self):
        df = pd.DataFrame(
            data={"Epoch": range(len(self.loss_history)), "Loss": self.loss_history}
        )

        plt.figure(figsize=(7, 5))
        sns.lineplot(data=df, x="Epoch", y="Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss (Cross-entropy)")
        plt.title("Loss vs Epoch")
        plt.grid(True)

        plt.show()

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
