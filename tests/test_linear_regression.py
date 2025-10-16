import numpy as np
from linear_regression_model.model import LinearRegression
import random

# y = 3x
X = np.linspace(0, 1000, num=800)
y = 3 * X + random.gauss(0, 1)

new_X = np.array([[40]])

model = LinearRegression(learning_rate=1e-6, epochs=2000)

model.fit(X=X, y=y)
model.parameters()
y_pred = model.predict(new_X)
print(y_pred)

model.view_loss()
