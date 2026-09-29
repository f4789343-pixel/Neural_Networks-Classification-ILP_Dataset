from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score
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
print(df.shape)

df['Gender'] = df['Gender'].map({'Male':0, 'Female':1})

df["Albumin_and_Globulin_Ratio"] = df["Albumin_and_Globulin_Ratio"].fillna(df["Albumin_and_Globulin_Ratio"].median())

X = df.drop(columns='Selector')
y = df['Selector']

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

y_train = (y_train == 2).astype(int)
y_test = (y_test == 2).astype(int)

model = MLPClassifier(hidden_layer_sizes=4,activation='relu', solver='adam',learning_rate_init=0.1,max_iter=10000,random_state=42)
model.fit(x_train,y_train)
predictions = model.predict(x_test)

accuracy = accuracy_score(y_test,predictions)
cm = confusion_matrix(y_test,predictions)
f1 = f1_score(y_test,predictions)
precision = precision_score(y_test,predictions)
recall = recall_score(y_test,predictions)

print('Accuracy:',accuracy)
print('Precision Score:',precision)
print('Recall Score:', recall)
print('Confusion Matrix:',cm)
print('F1 score:', f1)