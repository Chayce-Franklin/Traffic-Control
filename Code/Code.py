import numpy as np
import pandas as pd

data = pd.read_csv("traffic-prediction-dataset.csv")
print(data.describe)
print(data.head())
print(data.columns)

## Data Preprocessing 
from sklearn.model_selection import train_test_split

# Split Data into train and test

x_train, x_test, y_train, y_test = train_test_split(data[data.columns], test_size=0.2, random_state=0) 



from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()
print(scaler)