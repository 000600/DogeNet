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
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv('DOGE-USD.csv')
df = df[df['Close'].notna()]

# Scale values
scaler = MinMaxScaler(feature_range = (0, 1))
close = np.array(df[['Close']])
close = scaler.fit_transform(np.array(close))

# Divide training and testing sizes
percent_train = 0.75 # Training set percentage of total dataset
train_size = int(len(close) * percent_train)

# Create train and test datasets
close_train = close[:train_size]
close_test = close[train_size:]

# Get total number of closing prices within each dataset
total_prices_train = len(close_train)
total_prices_test = len(close_test)

# Create datasets
seq_len = 5 # Number of closing prices included in a single input

x_raw_train = []
y_raw_train = []
x_raw_test = []
y_raw_test = []

# Training data
for index in range(0, total_prices_train - seq_len, 1): # Loop through the list of closing prices and pair seq_len input closing values with one output closing value
  input = close_train[index: index + seq_len] # Get a list of seq_len prices to create the x value
  output = close_train[index + seq_len] # Get the next closing price for the y value

  # Add values to corresponding dataset lists
  x_raw_train.append(input)
  y_raw_train.append(output[0])

# Testing data
for index in range(0, total_prices_test - seq_len, 1): # Loop through the list of closing prices and pair seq_len input closing values with one output closing value
  input = close_test[index: index + seq_len] # Get a list of seq_len prices to create the x value
  output = close_test[index + seq_len] # Get the next closing price for the y value

  # Add values to corresponding dataset lists
  x_raw_test.append(input)
  y_raw_test.append(output[0])

# Get the number of patterns (unique input values)
n_patterns_train = len(x_raw_train)
n_patterns_test = len(x_raw_test)

# Reshape x-values
x_train = np.reshape(x_raw_train, (n_patterns_train, seq_len, 1))
x_test = np.reshape(x_raw_test, (n_patterns_test, seq_len, 1))

# Turn y-values into arrays
y_train = np.array(y_raw_train)
y_test = np.array(y_raw_test)

# Initialize optimizer
opt = Adam(learning_rate = 0.001)

# Get input shape
input_shape = (x_train.shape[1], x_train.shape[2])

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
history = model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size, validation_data = (x_test, y_test)) # To add early stopping, add 'callbacks = [early_stopping]'

# Visualize  loss and validation loss
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epoch_list = [i for i in range(epochs)]

plt.plot(epoch_list, loss, label = 'Loss')
plt.plot(epoch_list, val_loss, label = 'Validation Loss')
plt.title('Validation and Training Loss Across Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# View model's predictions compared to actual values
pred_train = model.predict(x_train) # Get model's predictions on test dataset
pred_test = model.predict(x_test) # Get model's predictions on train dataset

# Reformat pred_test so that it shows up correctly on the graph
filler = np.array([np.NaN for i in range(len(close_train))])
pred_test_filled = np.append(filler, pred_test)

# Visualize predictions and actual values
plt.plot(close, label = 'Actual Closing Price') # Plot actual values
plt.plot(pred_train, label = 'Predicted Closing Price (Train Data)') # Plot predictions on training data
plt.plot(pred_test_filled, label = 'Predicted Closing Price (Test Data)') # Plot predictions on testing data
plt.xlabel('Time (Index)')
plt.ylabel('Dogecoin Closing Price')
plt.title("Model's Predicted Closing Price Compared to Actual Closing Price of Dogecoin")
plt.legend()
plt.show()

# View prediction sequences

# Generate random index
index = np.random.randint(0, len(x_raw_test) - 1)

# Create seed
seed = x_test[index]
print("Seed:")
print(seed)
predictions = []
num_iterations = 100 # Change this number to have the model predict more values

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
seed = x_test[-1]
proj_predictions = []
num_iterations = 100 # Change this number to have the model predict more values (closing prices)

# Loop through num_iterations times and add model's prediction to inputs each time
for iter in range(num_iterations):
  input = np.reshape(seed, (1, len(seed), 1)) # Create input
  prediction = model.predict(input) # Get model's prediction
  proj_predictions.append(prediction[0])
  seed = np.append(seed, prediction) # Add model's prediction to seed so that it is taken into account in the next iteratior
  seed = seed[1: ] # Shift seed forward so that it maintains the correct shape (the shape is dictated by seq_len)

# Get total predictions on both training and testing sets
pred_total = np.append(pred_train, pred_test)

# Add filler values so that the projections appear in the right spot on the graph
filler = np.array([np.NaN for i in range(len(x_test) + len(x_train))])
projected = np.append(filler, proj_predictions)

# Visualize previous predictions and new projections
plt.plot(close, label = 'Actual Closing Price') # Plot actual values
plt.plot(pred_total, label = 'Predicted Closing Price') # Plot predictions
plt.plot(projected, label = 'Projected Closing Price') # Plot projections
plt.xlabel('Time (Index)')
plt.ylabel('Dogecoin Closing Price')
plt.title("Model's Projected Closing Price of Dogecoin")
plt.legend()
plt.show()
