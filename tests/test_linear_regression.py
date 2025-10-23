import numpy as np
import random
from sklearn.model_selection import train_test_split
from linear_regression_model import LinearRegression
from sklearn.linear_model import LinearRegression as NewLinearRegression

np.random.seed(42)
random.seed(42)

X = np.linspace(0, 1000, num=800).reshape(-1, 1)  # (800, 1)
y = 3 * X.flatten() + np.random.normal(0, 20, size=X.shape[0])  # linear + noise

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression(learning_rate=1e-2, epochs=3000)
model.fit(X=X_train, y=y_train)

model.parameters()


model.view_loss()

score = model.score(X_test, y_test)
print("R2 score on test set:", score)

# Use the model of sk-learn
better = NewLinearRegression()
better.fit(X_train, y_train)

r2 = better.score(X_test, y_test)
print(r2)
