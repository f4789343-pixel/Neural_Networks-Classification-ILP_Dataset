import pandas as pd
import numpy as np

columns = [
    "Age",
    "Gender",
    "Total_Bilirubin",
    "Direct_Bilirubin",
    "Alkaline_Phosphotase",
    "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase",
    "Total_Proteins",
    "Albumin",
    "Albumin_and_Globulin_Ratio",
    "Selector"
]

df = pd.read_csv('ilpd.zip',header=None, names=columns)
print(df.columns)

df['Gender'] = df['Gender'].map({'Male':0, 'Female':1})
print(df['Gender'].head())

X = df.drop(columns='target')
y = df['target']
np.random.seed(42)

indices = np.random.permutation(len(X))
test_size = int(len(X) * 0.2)

train_indices = indices[:test_size]
test_indices = indices[test_size:]

x_train = X.iloc[train_indices]
x_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]

mean = np.mean(x_train,axis=0)
std = np.std(x_train, axis=0)

x_train_scaled = 