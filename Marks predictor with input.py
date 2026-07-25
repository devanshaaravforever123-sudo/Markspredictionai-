import numpy as np
from sklearn.linear_model import LinearRegression

# Training data
# [study hours, sleep hours, practice tests, previous marks]

X = np.array([
    [2, 5, 1, 60],
    [4, 6, 2, 70],
    [6, 7, 4, 80],
    [8, 8, 6, 90],
    [10, 8, 8, 95]
])

y = np.array([62, 72, 82, 91, 98])

# Create and train model
model = LinearRegression()
model.fit(X, y)

# User inputs
study = float(input("Study hours: "))
sleep = float(input("Sleep hours: "))
tests = int(input("Practice tests: "))
previous = float(input("Previous marks: "))

# Prediction
prediction = model.predict([[study, sleep, tests, previous]])

print("Predicted marks:", round(prediction[0], 2))
