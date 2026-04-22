"""
Traffic Flow Prediction with Neural Networks(SAEs、LSTM、GRU).
"""
import sys
import json
import h5py
import tensorflow as tf
import tf_keras as keras
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

              # Recreate the model from its configuration
              model = keras.models.model_from_config(model_config)
        
          # Load the weights into the recreated model
          model.load_weights(model_path)
          loaded_models[model_name] = model
          print(f"Model '{model_name}' loaded successfully!")
          # Optional: Print model summary in main.py if needed
          # model.summary()

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
  _, _, X_test, y_test, scaler = process_data(file1, file2, lag)
  y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]

  y_preds = []
  for name, model in zip(model_names, models):
      if name.lower() == 'saes':
          X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
      else:
          X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
      file = 'images/' + name + '.png'
      plot_model(model, to_file=file, show_shapes=True)
      predicted = model.predict(X_test)
      predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
      y_preds.append(predicted[:288])
      print(name)
      eva_regress(y_test, predicted)

  plot_results(y_test[: 288], y_preds, model_names)


if __name__ == '__main__':
    main()
