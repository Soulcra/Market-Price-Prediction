import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset (assuming you have the data in a CSV file)
data = pd.read_csv("GE.csv")

# Convert the "Date" column to datetime format
data["Date"] = pd.to_datetime(data["Date"])

# Filter the data for the year 2020
mask = (data["Date"].dt.year == 2020)
filtered_data = data[mask]

# Create and train the Random Forest model with reduced complexity
n_estimators = 10  # Reduce the number of trees
max_depth = 5  # Decrease the max depth
min_samples_split = 5  # Increase the minimum samples required to split a node
random_state = 30

X = filtered_data[["Date"]]
y = filtered_data["Close"]

model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state)
model.fit(X, y)

# Make predictions for the entire dataset
X_pred = filtered_data[["Date"]]  # Use the same feature name as during training
y_pred = model.predict(X_pred)

# Calculate the mean squared error (MSE)
mse = mean_squared_error(filtered_data["Close"], y_pred)

# Plot the actual vs. predicted closing prices for the year 2020
plt.figure(figsize=(12, 6))
date_interval = 10  # Adjust the interval to control the spacing between dates
plt.plot(filtered_data["Date"].dt.strftime('%m/%d/%Y')[::date_interval], filtered_data["Close"][::date_interval], label="Actual Close Price", marker='o')
plt.plot(filtered_data["Date"].dt.strftime('%m/%d/%Y')[::date_interval], y_pred[::date_interval], label="Predicted Close Price", marker='x')
plt.xlabel("Date (MM/DD/YYYY)")
plt.ylabel("Close Price")
plt.title("Actual vs. Predicted Close Price ")
plt.xticks(rotation=45)
plt.legend()

# Add the MSE above the plot
plt.text(0.5, 0.95, f"MSE: {mse:.2f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

plt.show()
