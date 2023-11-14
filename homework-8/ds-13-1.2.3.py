from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

scatter_matrix = pd.plotting.scatter_matrix(data, c=df['target'], marker='o', s=10, hist_kwds={'bins': 20}, figsize=(10,10))
plt.suptitle('Scatter Matrix of Iris Dataset', size=16)
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(data, df['target'], random_state=0, test_size=0.2)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("训练集准确度:", train_accuracy)
print("测试集准确度:", test_accuracy)
