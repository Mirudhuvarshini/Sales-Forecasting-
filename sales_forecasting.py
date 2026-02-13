
# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load Dataset
dataset = pd.read_csv("Customer Purchasing Behaviors.csv")

# Convert 'region' column from categorical to numerical (if exists)
if "region" in dataset.columns:
    dataset["region"] = dataset["region"].str.strip()
    dataset["region"] = dataset["region"].map({
        "North": 0,
        "South": 1,
        "East": 2,
        "West": 3
    })
else:
    print("The 'region' column does not exist in the dataset.")

# Define Features (X) and Target (y)
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0
)

# Create Linear Regression Model
regressor = LinearRegression()

# Train Model
regressor.fit(X_train, y_train)

# Print Model Parameters
print("Intercept:", regressor.intercept_)
print("Coefficients:", regressor.coef_)

# Model Accuracy (R-squared)
score = regressor.score(X_train, y_train)
print("R-squared Score:", score)

# Predict on Training Data
y_pred = regressor.predict(X_train)

# Visualization (Using first feature column for plotting)
plt.figure(dpi=300)
plt.scatter(X_train[:, 0], y_train, color="blue", label="Actual Data")
plt.plot(X_train[:, 0], y_pred, color="purple", label="Regression Line")

plt.xlabel("Feature 1")
plt.ylabel("Target Value")
plt.title("Sales Forecasting using Linear Regression")
plt.legend()
plt.show()
