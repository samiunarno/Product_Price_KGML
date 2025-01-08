import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# Example Data (replace this with your actual dataset)
data = {
    'weight': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Potato weights in kilograms
    'price': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # Potato prices in Taka
}
df = pd.read_csv("PotatoPrice.csv")
print(df)

# Features and target variable
X = df[['KG']]  # Features (weight of potatoes)
y = df['price']  # Target (price of potatoes)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Regressor model
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train_scaled, y_train)

# Make a prediction
x_input = float(input('To know the potato price, enter the potato weight (in kilograms): '))
x_input_scaled = scaler.transform([[x_input]])  # Scale the input before prediction
predicted_price = reg.predict(x_input_scaled)

# Output the prediction
print(f'So, {x_input} kilograms of potatoes price is = {predicted_price[0]:.2f} Taka')

# Model evaluation
y_pred = reg.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae:.2f}')

