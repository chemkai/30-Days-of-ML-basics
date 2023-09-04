import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# loading the dataset to a Pandas DataFrame
dataset = pd.read_csv('Placement_Dataset.csv')
print(dataset.isnull().sum())

dataset['salary'].fillna(dataset['salary'].median(),inplace=True)
print(dataset.isnull().sum())