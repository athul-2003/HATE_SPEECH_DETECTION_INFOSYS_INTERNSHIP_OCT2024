#LINEAR REGRESSION
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

#  Load the dataset
data = pd.read_csv('labeled_data.csv')
print(data.head())

# Preprocess the data
# Define features (X) and target variable (y)
X = data.drop(columns=['class'])  # Input features
y = data['class']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the features (X) and the target (y)
X = data[['count', 'hate_speech', 'offensive_language', 'neither']]  # Select only numerical features
y = data['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the numerical features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

#  Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (MSE): {mse:.2f}')

r2 = r2_score(y_test, y_pred)
print(f'R² Score: {r2:.2f}')

#  Visualize the results (True vs Predicted values)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolors='black', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values')
plt.show()







--------
--------------
#*CONTACT*#
Name: BELLAMKONDA NAGA RAJU
Email: bellnagaraju3307@gmail.com