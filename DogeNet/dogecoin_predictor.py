# Imports
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
df = pd.read_csv('DOGE-USD.csv')
df = df[df['Close'].notna()]
close = np.array(df[['Close']])

# Get total number of prices in the data
total_prices = len(close)

# Create dataset
seq_len = 5 # Number of closing prices included in a single input

x_raw = []
y_raw = []

# Loop through the list of closing prices and pair seq_len input closing values with one output closing value
for index in range(0, total_prices - seq_len, 1):
  input = close[index: index + seq_len] # Get a list of seq_len prices to create the x value
  output = close[index + seq_len] # Get the next closing price for the y value

  # Add values to corresponding dataset lists
  x_raw.append(input)
  y_raw.append(output[0])

# Get the number of patterns (unique input values)
n_patterns = len(x_raw)

# Reshape x-values
x = np.reshape(x_raw, (n_patterns, seq_len, 1))

# Turn y values into arrays
y = np.array(y_raw)

# Initialize optimizer
opt = Adam(learning_rate = 0.001)

# Get input shape
input_shape = (x.shape[1], x.shape[2])

# Create model
model = Sequential()

# Input LSTM layer
model.add(LSTM(50, input_shape = input_shape, return_sequences = True, activation = 'tanh'))
model.add(Dropout(0.2))

# Hidden layers
model.add(LSTM(50, activation = 'tanh'))
model.add(Dropout(0.2))

# Output layer
model.add(Dense(1))

# Configure early stopping
early_stopping = EarlyStopping(min_delta = 0.001, patience = 10, restore_best_weights = True)

# Set epochs and batch size
epochs = 50
batch_size = 64

# Compile and train model
model.compile(optimizer = opt, loss = 'mse')
history = model.fit(x, y, epochs = epochs, batch_size = batch_size) # To add early stopping, add 'callbacks = [early_stopping]'

# View model's predictions compared to actual values

# Get model's predictions
pred = model.predict(x)

# Visualize predictions and actual values
plt.plot(close, label = 'Actual Closing Price') # Plot actual values
plt.plot(pred, label = 'Predicted Closing Price') # Plot predictions
plt.xlabel('Time (Index)')
plt.ylabel('Dogecoin Closing Price')
plt.title("Model's Predicted Closing Price Compared to Actual Closing Price of Dogecoin")
plt.legend()
plt.show()

# View prediction sequences

# Generate random index
index = np.random.randint(0, len(x_raw) - 1)

# Create seed
seed = x[index]
print("Seed:")
print(seed)
predictions = []
num_iterations = 100 # Change this number to have the model predict more values (closing prices)

# Loop through num_iterations times and add model's prediction to inputs each time
for iter in range(num_iterations):
  input = np.reshape(seed, (1, len(seed), 1)) # Create input
  prediction = model.predict(input) # Get model's prediction
  predictions.append(prediction[0])
  seed = np.append(seed, prediction) # Add model's prediction to seed so that it is taken into account in the next iteratior
  seed = seed[1: ] # Shift seed forward so that it maintains the correct shape (the shape is dictated by seq_len)

# Visualize model's predictions
plt.plot(predictions)
plt.xlabel('Time (Index)')
plt.ylabel('Projected Price')
plt.title("Model's Projected Closing Price of Dogecoin Over Time")
plt.show()

# View model's predicted outlook on Dogecoin after the dataset

# Create seed
seed = x[-1]
proj_predictions = []
num_iterations = 100 # Change this number to have the model predict more values (closing prices)

# Loop through num_iterations times and add model's prediction to inputs each time
for iter in range(num_iterations):
  input = np.reshape(seed, (1, len(seed), 1)) # Create input
  prediction = model.predict(input) # Get model's prediction
  proj_predictions.append(prediction[0])
  seed = np.append(seed, prediction) # Add model's prediction to seed so that it is taken into account in the next iteratior
  seed = seed[1: ] # Shift seed forward so that it maintains the correct shape (the shape is dictated by seq_len)

# Add filler values so that the projections appear in the right spot on the graph
filler = np.array([np.NaN for i in range(len(x))])
projected = np.append(filler, proj_predictions)

# Visualize previous predictions and new projections
plt.plot(close, label = 'Actual Closing Price') # Plot actual values
plt.plot(pred, label = 'Predicted Closing Price') # Plot predictions
plt.plot(projected, label = 'Projected Closing Price') # Plot projections
plt.xlabel('Time (Index)')
plt.ylabel('Dogecoin Closing Price')
plt.title("Model's Projected Closing Price of Dogecoin")
plt.legend()
plt.show()
