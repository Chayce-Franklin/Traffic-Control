"""
Traffic Flow Prediction with Neural Networks(SAEs、LSTM、GRU).
"""
import sys
import json
import h5py
import tensorflow as tf
from tensorflow import keras
import math
import warnings
import numpy as np
import pandas as pd
from data.data import process_data
from keras.models import load_model
from tensorflow.keras.utils import plot_model
import sklearn.metrics as metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.ion()

def MAPE(y_true, y_pred):
    """Mean Absolute Percentage Error
    Calculate the mape.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    # Returns
        mape: Double, result data for train.
    """

    y = [x for x in y_true if x > 0]
    y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]

    num = len(y_pred)
    sums = 0

    for i in range(num):
        tmp = abs(y[i] - y_pred[i]) / y[i]
        sums += tmp

    mape = sums * (100 / num)

    return mape


def eva_regress(y_true, y_pred):
    """Evaluation
    evaluate the predicted resul.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    """

    mape = MAPE(y_true, y_pred)
    vs = metrics.explained_variance_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    print('explained_variance_score:%f' % vs)
    print('mape:%f%%' % mape)
    print('mae:%f' % mae)
    print('mse:%f' % mse)
    print('rmse:%f' % math.sqrt(mse))
    print('r2:%f' % r2)


def plot_results(y_true, y_preds, names):
    """Plot
    Plot the true data and predicted data.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
        names: List, Method names.
    """
    d = '2016-3-4 00:00'
    x = pd.date_range(d, periods=288, freq='5min')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, y_true, label='True Data')
    for name, y_pred in zip(names, y_preds):
        ax.plot(x, y_pred, label=name)

    plt.legend()
    plt.grid(True)
    plt.xlabel('Time of Day')
    plt.ylabel('Flow')

    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    plt.show()
class NoisyDense(keras.layers.Layer):
    """Dense layer with parametric noise: w = mu + epsilon * xi"""

    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        n_in = int(input_shape[-1])
        self.mu_kernel = self.add_weight(
            shape=(n_in, self.units), name='mu_kernel',
            initializer='glorot_uniform', trainable=True)
        self.mu_bias = self.add_weight(
            shape=(self.units,), name='mu_bias',
            initializer='zeros', trainable=True)
        self.epsilon_kernel = self.add_weight(
            shape=(n_in, self.units), name='epsilon_kernel',
            initializer=keras.initializers.Constant(0.017), trainable=True)
        self.epsilon_bias = self.add_weight(
            shape=(self.units,), name='epsilon_bias',
            initializer=keras.initializers.Constant(0.017), trainable=True)

    def call(self, inputs, training=None):
        if training:
            xi_kernel = tf.random.normal(shape=tf.shape(self.mu_kernel))
            xi_bias   = tf.random.normal(shape=tf.shape(self.mu_bias))
            w = self.mu_kernel + self.epsilon_kernel * xi_kernel
            b = self.mu_bias   + self.epsilon_bias   * xi_bias
        else:
            w = self.mu_kernel
            b = self.mu_bias
        out = tf.matmul(inputs, w) + b
        return self.activation(out) if self.activation else out

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units,
                       'activation': keras.activations.serialize(self.activation)})
        return config
def make_noisy_copy(original_model, model_name):
    """Rebuild model with NoisyDense output layer, copying pretrained weights."""

    if model_name == 'lstm':
        noisy_model = keras.Sequential([
            keras.layers.LSTM(64, input_shape=(12, 1), return_sequences=True,
                              activation='tanh', recurrent_activation='hard_sigmoid',
                              name="lstm_1"),
            keras.layers.LSTM(64, return_sequences=False,
                              activation='tanh', recurrent_activation='hard_sigmoid',
                              name="lstm_2"),
            keras.layers.Dropout(0.2, name="dropout_1"),
            NoisyDense(1, activation="sigmoid", name="noisy_output")
        ])
        noisy_model.build(input_shape=(None, 12, 1))

    elif model_name == 'gru':
        noisy_model = keras.Sequential([
            keras.layers.GRU(64, input_shape=(12, 1), return_sequences=True, name="gru"),
            keras.layers.GRU(64, return_sequences=True, name="gru_1"),
            keras.layers.Flatten(name="flatten"),
            NoisyDense(1, activation="linear", name="noisy_output")
        ])
        noisy_model.build(input_shape=(None, 12, 1))

    elif model_name == 'saes':
        noisy_model = keras.Sequential([
            keras.layers.Dense(400, activation="linear", input_shape=(12,), name="hidden1"),
            keras.layers.Activation("sigmoid", name="activation_4"),
            keras.layers.Dense(400, activation="linear", name="hidden2"),
            keras.layers.Activation("sigmoid", name="activation_5"),
            keras.layers.Dense(400, activation="linear", name="hidden3"),
            keras.layers.Activation("sigmoid", name="activation_6"),
            keras.layers.Dropout(0.2, name="dropout_4"),
            NoisyDense(1, activation="sigmoid", name="noisy_output")
        ])
        noisy_model.build(input_shape=(None, 12))

    # Copy all shared layer weights (everything except the final Dense)
    for orig_layer, noisy_layer in zip(original_model.layers[:-1], noisy_model.layers[:-1]):
        noisy_layer.set_weights(orig_layer.get_weights())

    # Copy old Dense weights into mu, leave epsilon small
    old_kernel, old_bias = original_model.layers[-1].get_weights()
    noisy_out = noisy_model.get_layer("noisy_output")
    noisy_out.set_weights([
        old_kernel,
        old_bias,
        np.full_like(old_kernel, 0.017),
        np.full_like(old_bias,   0.017)
    ])

    return noisy_model
def main():
  model_names = ['lstm', 'gru', 'saes']
  loaded_models = {}

  for model_name in model_names:
      model_path = f'model/{model_name}.h5' # Adjust this path if main.py's current directory is different from where the model is located
    
      try:
          with h5py.File(model_path, 'r') as f:
              model_config_str = f.attrs.get('model_config')

              if model_config_str is None:
                  raise ValueError(f"model_config attribute not found in the HDF5 file for {model_name}.h5.")

              model_config = json.loads(model_config_str)

              #if model_name == 'gru':
                # Iterate through layers and explicitly set reset_after for GRU layers
                #if 'config' in model_config and isinstance(model_config['config'], list):
                    #for layer_config in model_config['config']:
                        #if layer_config.get('class_name') == 'GRU':
                            #if 'config' in layer_config and 'reset_after' not in layer_config['config']:
                                #layer_config['config']['reset_after'] = False
                                #print(f"  Explicitly set 'reset_after: False' for GRU layer in {model_name}.h5 config.")

          from tensorflow import keras
          try:
            # Try to load the full model (skip optimizer state to avoid Keras 3 issues)
            model = keras.models.load_model(model_path, compile=False)
          except Exception as e:
            print(f"Standard load failed for {model_name}: {e}")
            print(f"Attempting to rebuild architecture for {model_name} and load only weights...")

            # Import your architecture builders here
            from tensorflow import keras

            if model_name.lower() == "lstm":
                model = keras.Sequential([
                    keras.layers.LSTM(64, input_shape=(12, 1), return_sequences=True,
                          activation='tanh', recurrent_activation='hard_sigmoid',
                          name="lstm_1"),
                    keras.layers.LSTM(64, return_sequences=False,
                          activation='tanh', recurrent_activation='hard_sigmoid',
                          name="lstm_2"),
                    keras.layers.Dropout(0.2, name="dropout_1"),
                    keras.layers.Dense(1, activation='sigmoid', name="dense_1")
                ])
            elif model_name.lower() == "gru":
                model = keras.Sequential([
                    keras.layers.GRU(64, input_shape=(12, 1), return_sequences=True, name="gru"),
                    keras.layers.GRU(64, return_sequences=True, name="gru_1"),
                    keras.layers.Flatten(name="flatten"),
                    keras.layers.Dense(1, activation="linear", name="dense")
                ])
            elif model_name.lower() == "saes":
                model = keras.Sequential([
                    keras.layers.Dense(400, activation="linear", input_shape=(12,), name="hidden1"),
                    keras.layers.Activation("sigmoid", name="activation_4"),
                    keras.layers.Dense(400, activation="linear", name="hidden2"),
                    keras.layers.Activation("sigmoid", name="activation_5"),
                    keras.layers.Dense(400, activation="linear", name="hidden3"),
                    keras.layers.Activation("sigmoid", name="activation_6"),
                    keras.layers.Dropout(0.2, name="dropout_4"),
                    keras.layers.Dense(1, activation="sigmoid", name="dense_4")
                ])
            else:
                raise ValueError(f"Unknown model type: {model_name}")

            model.load_weights(model_path)
            print(f"Weights loaded successfully for {model_name}.")

          except Exception as inner_e:
                print(f"Fallback load also failed for {model_name}: {inner_e}")
                sys.exit(1)
          loaded_models[model_name] = model
          print(f"Model '{model_name}' loaded successfully!")

      except Exception as e:
           print(f"Error loading model '{model_name}' manually in main.py: {e}")
          # Handle errors, e.g., exit if a critical model can't be loaded
           sys.exit(1)

  lstm = loaded_models['lstm']
  gru = loaded_models['gru']
  saes = loaded_models['saes']

  models = [lstm, gru, saes]

  lag = 12
  file1 = 'data/train.csv'
  file2 = 'data/test.csv'
  X_train, y_train, X_test, y_test, scaler = process_data(file1, file2, lag)  # X_test defined here
  y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]
  X_test_flat = X_test.reshape(X_test.shape[0], -1)          # (N, 12) for SAES
  X_test_seq  = X_test_flat.reshape(X_test_flat.shape[0], 12, 1)  # (N, 12, 1) for LSTM/GRU

  y_preds = []

  for name, current_model in zip(model_names, models):
        X_input = X_test_flat if name == 'saes' else X_test_seq  # never mutate X_test!

        #img_path = f'images/{name}.png'
        #plot_model(current_model, to_file=img_path, show_shapes=True)

        predicted = current_model.predict(X_input)
        predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
        y_preds.append(predicted[:288])
        print(name)
        eva_regress(y_test, predicted)

  plot_results(y_test[: 288], y_preds, model_names)
  print("\n--- Noisy Models ---")
  y_preds_noisy = []
  X_train_flat = X_train.reshape(X_train.shape[0], -1)
  X_train_seq  = X_train_flat.reshape(X_train_flat.shape[0], 12, 1)
  y_train_inv  = scaler.inverse_transform(y_train.reshape(-1, 1)).reshape(1, -1)[0]

  for name in model_names:
      noisy = make_noisy_copy(loaded_models[name], name)
      X_train_input = X_train_flat if name == 'saes' else X_train_seq
      X_test_input  = X_test_flat  if name == 'saes' else X_test_seq

      # Retrain with noise on training data
      noisy.compile(optimizer='adam', loss='mse')
      noisy.fit(X_train_input, y_train, epochs=10, batch_size=32, verbose=1)

      # Evaluate on test data 
      predicted = noisy.predict(X_test_input)
      predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
      y_preds_noisy.append(predicted[:288])
      print(f"{name} (noisy)")
      eva_regress(y_test, predicted)

  plot_results(y_test[:288], y_preds_noisy, [f"{n}_noisy" for n in model_names])
  input("Hit enter to end program")

if __name__ == '__main__':
    main()
