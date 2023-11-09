from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

dt = load_iris()
X_train, X_test, Y_train, Y_test = train_test_split(dt['data'], dt['target'], shuffle=True, test_size=0.3)

standard = StandardScaler()
X_train = standard.fit_transform(X_train)
X_test = standard.transform(X_test)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, Y_train)
Y_pred = lr_model.predict(X_test)
print("Logistic回归分类预测结果：")
print(Y_pred)

df = pd.DataFrame(data=dt.data, columns=dt.feature_names)
print("各特征值的均值：")
print(df.mean())
euclidean = lambda row: np.linalg.norm(row - df.mean())
df['dist'] = df.apply(euclidean, axis=1)
print("各样本数据点到中心点的欧氏距离：")
print(df)

k = 3
df = df[['petal length (cm)', 'petal width (cm)']]
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(df)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

plt.scatter(df['petal length (cm)'], df['petal width (cm)'], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', label='Centroids')
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.legend(loc='upper left')
plt.title("KMeans Illustration")
plt.tight_layout()
plt.show()
