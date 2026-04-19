# Data Libraries
import numpy as np
import pandas as pd

# Data Scrape package
import pandas_datareader.data as web

# Plotting Package
import matplotlib.pyplot as plt

# Scaling Package
from sklearn.model_selection import train_test_split

# Keras 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

## Data Preprocessing 

data = pd.read_csv("traffic-prediction-dataset.csv")
print(data.describe)
print(data.head())
print(data.columns)


# Split Data into train and test

x_train, x_test, y_train, y_test = train_test_split(data[data.columns], test_size=0.2, random_state=0) 



from sklearn.preprocessing import MinMaxScaler

min_max_sc = MinMaxScaler(feature_range=(0,1))
X_train = min_max_sc.fit_transform(x_train)
X_test = min_max_sc.transform(x_test)
df = pd.DataFrame(X_train)

## Recurrent Nueral Networks

# GRU
model = keras.Sequential()

# Add GRU Layer
model.add(layers.GRU(3,
                     activation = 'tanh',
                     recurrent_activation = "sigmoid",
                     input_shape=(X_train.shape[1], X_train.shape[2])))

# Add a dropout layer that prevents overfitting

model.add(layers.Dropout(rate=0.2))

# Add a Dense layer 
model.add(layers.Dense(1))

# Evaluate Loss function of MSE usin the adam optimizer
model.compile(loss='mean_sqaured_error', optimizer = 'adam')

# print out architecture
model.summary()

# Fitting the data
history = model.fit(X[:threshold],
                    Y[:threshold],
                    shuffle = False,
                    epochs=50,
                    batch_size=128,
                    validation_split=0.05,
                    verbose=1)




