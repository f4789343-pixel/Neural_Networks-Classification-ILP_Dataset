import matplotlib.pyplot as plt
from scikit_learn import y_test, cm
import numpy as np

classes = np.unique(y_test)

plt.imshow(cm)
plt.xlabel('Predicted')
plt.ylabel('actual')
plt.title('Confusion Matrix - Neural Networks')
plt.xticks(range(len(classes)), classes)
plt.yticks(range(len(classes)), classes)

for i in range(len(classes)):
  for j in range(len(classes)):
    plt.text(i,j,cm[i][j])
plt.colorbar()
plt.savefig('confusion_matrix.png')
plt.show()
