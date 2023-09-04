import numpy as numpy
import pandas as pd
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

dataset = sklearn.datasets.load_breast_cancer()

#Loading the datasets to the pandas dataframe
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

X = df
Y = dataset.target

# Splitting the data into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Standardizing the data
scaler = StandardScaler()
#calculating the mean and median using later
scaler.fit(X_train)

x_train_standardized = scaler.transform(X_train)

print(x_train_standardized)