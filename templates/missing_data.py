from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

dataset = pd.read_csv("data/Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])