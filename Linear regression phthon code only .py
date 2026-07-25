from sklearn.linear_model import LinearRegression
import numpy as np

# Training data
# [study hours, sleep hours, practice tests, previous marks]
X = np.array([
    [2, 5, 1, 60],
    [4, 6, 2, 70],
    [6, 7, 4, 80],
    [8, 8, 6, 90],
    [10, 8, 8, 95]
])

# Actual marks
y = np.array([62, 72, 82, 91, 98])

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict
prediction = model.predict([[7, 7, 5, 85]])

print("Predicted marks:", prediction[0])
