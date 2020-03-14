import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import pandas as pd


class KMeans:
    def __init__(self, k):
        self._k = k  # 聚类簇数
        self.centroids = None  # 聚类中心

        # 聚类标签，第一列为样本聚类结果的中心，第二列为样本和聚类中心的距离
        self.clusterLabel = None

    def _euclDistance(self, sample, centroid):
        """  计算一个样本和所有聚类中心的距离 """
        return np.sum((sample - centroid) ** 2, axis=1)

    def fit(self, X):
        numSamples = X.shape[0]
        self.centroids = X[np.random.choice(numSamples, self._k), :]  # 随机采样
        self.clusterLabel = np.zeros((numSamples, 2))

        clusterChanged = True  # 标记聚类中心是否改变

        # 循环，直到聚类中心不再改变
        while clusterChanged:
            clusterChanged = False

            # 遍历每个样本
            for i in range(numSamples):
                # 计算样本和每个聚类中心的距离，并找到最小距离
                distances = self._euclDistance(X[i, :], self.centroids)
                minIndex = np.argmin(distances)

                # 更新样本的聚类结果
                if self.clusterLabel[i, 0] != minIndex:
                    self.clusterLabel[i, :] = minIndex, distances[minIndex]

            # 更新聚类中心
            for i in range(self._k):
                # 取出第 i 蔟 的所有点
                pointsInCluster = X[np.nonzero(self.clusterLabel[:, 0] == i)[0]]
                # 新的聚类中心
                new_centroid = np.mean(pointsInCluster, axis=0)
                # 若聚类中心改变，进行更新
                if (self.centroids[i, :] != new_centroid).all():
                    clusterChanged = True
                    self.centroids[i, :] = new_centroid

    def predict(self, X_test):
        y_pred = np.zeros(len(X_test))
        for i in range(len(X_test)):
            distances = self._euclDistance(X_test[i, :], self.centroids)
            y_pred[i] = np.argmin(distances)

        return y_pred


# 测试

# 制作数据集
X, y = make_blobs(n_samples=500, n_features=2, centers=4, random_state=100)

# 展示数据
color = ["red", "pink", "orange", "gray"]
fig, ax = plt.subplots(1)
for i in range(4):
    ax.scatter(X[y == i, 0], X[y == i, 1], marker='o', s=8, c=color[i])
plt.show()

kmeans = KMeans(4)
kmeans.fit(X)
centroids = kmeans.centroids

# 展示结果
color = ["red", "pink", "orange", "gray"]
fig, ax = plt.subplots(1)
for i in range(4):
    ax.scatter(X[y == i, 0], X[y == i, 1], marker='o', s=8, c=color[i])

ax.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=50, c='black')

plt.show()
