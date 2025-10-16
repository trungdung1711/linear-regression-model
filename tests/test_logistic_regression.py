import numpy as np
from linear_regression_model import LogisticRegression
from sklearn.metrics import classification_report

np.random.seed(42)

N = 100

# Generate one feature X (values around 0)
X = np.random.randn(N, 1)

# Define the true relationship
# True relationship: y = 1 if (2*X + noise > 0)
y = (2 * X + 0.3 * np.random.randn(N, 1) > 0).astype(int)


# TEST SET
np.random.seed(123)  # separate seed for test set

# Number of test samples
N_test = 20

# Generate test feature
X_test = np.random.randn(N_test, 1)

# True labels (same rule as training)
y_test = (2 * X_test + 0.3 * np.random.randn(N_test, 1) > 0).astype(int)

model = LogisticRegression(learning_rate=1e-6, epochs=3000)

model.fit(X, y)
model.view_loss()

y_pred = model.predict(X_test)
report = classification_report(y_true=y_test, y_pred=y_pred)
print(report)
