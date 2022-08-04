# DogeNet

## The Neural Networks
These recurrent neural networks predict the closing price of Dogecoin based on the five most recent previous closing prices; the models take an input consist of a numpy array with the structure [[*closing price five days ago*, *closing price four days ago*, *closing price three days ago*, *closing price two days ago*, *closing price one day ago*]] and output a predicted closing price based on that input. Since both models try to predict closing price values close to actual values, they each use a mean squared error loss function and have 1 output neuron (since they are only predicting one output value — the closing price). They use a standard Adam optimizer with a learning rate of 0.001.

1. The first model, found in the **dogecoin_predictor.py** file, is a RNN that uses all of the available data and can predict the closing price well up until the data finishes, after which it can only predict reasonable closing prices somewhat well. It contains an architecture consisting of:
    - 1 Input LSTM layer (with 50 neurons, a standard tanh activation function, and an input shape of (5, 1))
    - 1 Dropout layer (with a dropout rate of 0.2)
    - 1 LSTM layer (with 50 units and a standard tanh activation function)
    - 1 Dropout layer (with a dropout rate of 0.2)
    - 1 Output layer (with 1 neuron and no activation function)

2. The second model, found in the **dogecoin_forecaster.py** file, is a RNN that uses some of the available data as training data and the rest as testing data. It can predict the Dogecoin closing price reasonably well with inputs within its training data, and is more adept at predicting reasonable closing prices on data outside of the provided training dataset. It has an architecture consisting of:
    - 1 Input LSTM layer (with 50 neurons, a standard tanh activation function, and an input shape of (5, 1))
    - 1 Dropout layer (with a dropout rate of 0.2)
    - 1 LSTM layer (with 50 units and a standard tanh activation function)
    - 1 Dropout layer (with a dropout rate of 0.2)
    - 1 Output layer (with 1 neuron and no activation function)

Feel free to further tune the hyperparameters or build upon either of the models!

## The Dataset
The dataset used here can be found at this link: https://www.kaggle.com/datasets/neelgajare/dogecoin-historical-price-data. Credit for the dataset collection goes to **Ruthwik333**, **Neel Gajare**, **Ifeanyi Chigbo**, and others on *Kaggle*. The dataset contains daily data on Dogecoin from November 9, 2017, to July 28, 2022. The attributes found in the data are:

- Daily opening price
- Daily closing price
- Daily volume
- Daily high
- Daily low
- Daily adjacent close

Note that in the **dogecoin_forecaster.py** file the data is preprocessed with Scikit-Learn's **MinMaxScaler()**.

## Libraries
These neural networks were created with the help of the Tensorflow and Scikit-Learn libraries.
- Tensorflow's Website: https://www.tensorflow.org/
- Tensorflow Installation Instructions: https://www.tensorflow.org/install
- Scikit-Learn's Website: https://scikit-learn.org/stable/
- Scikit-Learn's Installation Instructions: https://scikit-learn.org/stable/install.html

## Disclaimer
Please note that I do not recommend, endorse, or encourage the use of any of my work here in actual financial use or application in any way.
