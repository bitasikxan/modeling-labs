import os
os.environ["OMP_NUM_THREADS"] = "2"
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler


path = "C:\\Users\\legen\\Desktop\\model\\lab 5\\vowel.csv"
dataset = pd.read_csv(path)

X = dataset.drop('Class', axis=1)

# num of classes = 11
clusterer = KMeans(n_clusters=11, n_init=10, random_state=52)
clusterer.fit(X)
predictions = clusterer.predict(X)
centroids = clusterer.cluster_centers_

feature_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
column_names = X.columns

fig, axes = plt.subplots(3, 2, figsize=(15, 9))
axes = axes.ravel()  # для можливості ітерації

for i, (x_idx, y_idx) in enumerate(feature_pairs):
    ax = axes[i]

    scatter = ax.scatter(X.iloc[:, x_idx], X.iloc[:, y_idx],
                         c=predictions, cmap='tab20', s=20, alpha=0.6)

    ax.scatter(centroids[:, x_idx], centroids[:, y_idx],
               c='black', marker='x', s=200, linewidths=3, label='Centroids')

    ax.set_xlabel(f"{column_names[x_idx]}")
    ax.set_ylabel(f"{column_names[y_idx]}")
    ax.set_title(f"Cluster analysis: {column_names[x_idx]} vs {column_names[y_idx]}")
    ax.legend()

axes[5].axis('off')
plt.tight_layout()
plt.show()

dataset['cluster'] = predictions
print("Результати:")
print(dataset, "\n")
count_cluster = Counter(clusterer.labels_.astype(int))
print("К-сть у кластерах:")
print(count_cluster)



