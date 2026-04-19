"""
Defination of NN model
"""
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Activation, Flatten
import tensorflow.keras as keras


def get_lstm(units):
    """LSTM(Long Short-Term Memory)
    Build LSTM Model.

    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential([
        LSTM(units[1], input_shape=(units[0], 1), return_sequences=True),
        LSTM(units[2], return_sequences=False),
        Dense(units[3], activation='linear')
    ])
    return model


def get_gru(units):
    """GRU(Gated Recurrent Unit)
    Build GRU Model.

    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """
    model = Sequential([
        GRU(units[1], input_shape=(units[0], 1), return_sequences=True),
        GRU(units[2], return_sequences=True),
        Flatten(),
        Dense(units[3], activation='linear')
    ])
    return model
    


def _get_saes(units):
    """SAE(Auto-Encoders)
    Build SAE Model.

    # Arguments
        inputs: Integer, number of input units.
        hidden: Integer, number of hidden units.
        output: Integer, number of output units.
    # Returns
        model: Model, nn model.
    """
    model = Sequential([
        Dense(units[1], input_shape=(units[0],), activation='relu'),
        Dense(units[2], activation='relu'),
        Dense(units[3], activation='linear')
    ])
    return model
    


def get_da_rnn(T, D, P):
    # The Da-RNN model architecture would be defined here.
    # For simplicity, returning a placeholder or simplified model.
    # This part would need the full Da-RNN implementation.
    model = Sequential([
        Dense(P, input_shape=(T, D), activation='relu'),
        LSTM(P, return_sequences=True),
        LSTM(P),
        Dense(1, activation='linear')
    ])
    return model
