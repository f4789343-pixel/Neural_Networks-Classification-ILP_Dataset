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
print(df["Albumin_and_Globulin_Ratio"].head())
df['Gender'] = df['Gender'].map({'Male':0, 'Female':1})
print(df['Gender'].head())
df["Albumin_and_Globulin_Ratio"] = df["Albumin_and_Globulin_Ratio"].fillna(df["Albumin_and_Globulin_Ratio"].median())
print(df.isnull().sum())
X = df.drop(columns='Selector')
y = df['Selector']
np.random.seed(42)

indices = np.random.permutation(len(X))
test_size = int(len(X) * 0.2)

train_indices = indices[test_size:]
test_indices = indices[:test_size]

x_train = X.iloc[train_indices]
x_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]
print(y_train.head(10))
x_train = x_train.to_numpy()
x_test = x_test.to_numpy()


y_train = (y_train == 2).astype(int)
y_test  = (y_test == 2).astype(int)

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

mean = np.mean(x_train,axis=0)
std = np.std(x_train, axis=0)

x_train_scaled = (x_train - mean) / std
x_test_scaled = (x_test - mean) / std
#print(x_train_scaled)
#print(x_train.shape)
#print(x_test.shape)
w1 = np.random.randn(4, len(x_train[0]))
b1 = np.random.randn(4)

w2 = np.random.randn(1, 4)
b2 = np.zeros(1)
lr = 0.0001
def ReLU(z):
  return np.maximum(0,z)

def sigmoid(z):
  return 1 / (1+np.exp(-z))

for epoch in range(5000):
  for i in range(len(x_train)):
    z = w1 @ x_train_scaled[i] + b1
    #print(np.max(np.abs(z)))
    a = ReLU(z)
    y_pred = w2 @ a + b2
    a2 = sigmoid(y_pred)
    loss = -(y_train[i] * np.log(a2) + (1 - y_train[i])*np.log(1-a2))
    error = a2 - y_train[i]

    dw2 = error[:, None] * a[None,:]
    db2 = error

    delta = (w2.T @ error) * (z > 0).astype(float)
    dw1 = delta[:,None] * x_train_scaled[i][None, :]
    db1 = delta

    w2 -= lr*dw2
    b2 -= lr*db2

    w1 -= lr*dw1
    b1 -= lr*db1
pred = []
for i in range(len(x_test_scaled)):
   z = w1 @ x_test_scaled[i] + b1
   a = ReLU(z)
   y_pred = w2 @ a + b2
   a2 = sigmoid(y_pred)
   pred.append(a2)

print(pred)
probs = []
for i in pred:
  if i >= 0.5:
    probs.append(1)
  else:
    probs.append(0)
print(probs)

c = 0
for i in range(len(y_test)):
  if probs[i] == y_test[i]:
    c += 1
accuracy = c / len(y_test)
print('Accuracy:',accuracy)

tp = 0
tn = 0
fp = 0
fn = 0
for actual,predicted in zip(y_test,probs):
  if actual == 1 and predicted == 1:
    tp += 1
  elif actual == 0 and predicted == 0:
    tn += 1
  elif actual == 0 and predicted == 1:
    fp += 1
  else:
    fn += 1

print('TP:', tp)
print('TN:', tn)
print('FP:', fp)
print('FN:',fn)

precision = tp / (tp+fp)
recall = tp / (tp+fn)
f1 = 2*(precision*recall) / (precision+recall)

def confusion_matrix(y_test,predictions):
  classes = np.unique(y_test)
  matrix = np.zeros((len(classes),len(classes)),dtype=int)
  for actual, predicted in zip(y_test,predictions):
    actual_index = np.where(actual == classes)[0][0]
    predicted_index = np.where(predicted == classes)[0][0]
    matrix[actual_index][predicted_index] += 1
  return matrix
cm = confusion_matrix(y_test,probs)

print('Precision Score:', precision)
print('Recall Score:', recall)
print('Confusion Matrix:', cm)
  


 
