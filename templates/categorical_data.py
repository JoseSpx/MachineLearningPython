import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import make_column_transformer

dataset = pd.read_csv("data/Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

A = make_column_transformer((OneHotEncoder(categories='auto'), [0]), remainder="passthrough")
x = A.fit_transform(x)

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
y = np.reshape(y, (-1, 1))