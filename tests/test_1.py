import numpy as np
from linear_regression_model import LinearRegression
import random

# y = 3x
X = np.linspace(0, 100, num=500)
y = 3 * X + random.gauss(0, 1)

new_X = np.array([[43]])

model = LinearRegression()

model.fit(X=X, y=y)
model.parameters()
y_pred = model.predict(new_X)
print(y_pred)
