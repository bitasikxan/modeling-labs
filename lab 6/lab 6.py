import os
os.environ["OMP_NUM_THREADS"] = "2"
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tabulate import tabulate
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

labels = clusterer.labels_.tolist()
dataset['cluster'] = predictions

cluster_content = dataset.groupby(["cluster", "Class"]).size().unstack(fill_value=0)
cluster_content['Total'] = cluster_content.sum(axis=1)
cluster_content.loc['Total'] = cluster_content.sum()

print(tabulate(cluster_content, headers="keys", tablefmt="psql"))
count_cluster = Counter(labels)

print("К-сть у кластерах:")
print(count_cluster)

#================================================================================================

n_range = range(2, 34)
inertia = []
silhouette = []
davies_bouldin = []
calinski_harabasz = []

for n in n_range:
    clusterer = KMeans(n_clusters=n, n_init=10, random_state=52)
    clusterer.fit(X)
    labels = clusterer.labels_

    inertia.append(clusterer.inertia_)
    silhouette.append(metrics.silhouette_score(X, labels))
    davies_bouldin.append(metrics.davies_bouldin_score(X, labels))
    calinski_harabasz.append(metrics.calinski_harabasz_score(X, labels))

metrics_df = pd.DataFrame({
    'Clusters': n_range,
    'Inertia': inertia,
    'Silhouette': silhouette,
    'Davies-Bouldin': davies_bouldin,
    'Calinski-Harabasz': calinski_harabasz
})

print("\n=== Таблиця метрик ===")
print(metrics_df)


fig, axes = plt.subplots(2, 2, figsize=(15, 9))
axes = axes.ravel()

axes[0].plot(n_range, inertia, marker='o', color='blue')
axes[0].set_title('Inertia (метод ліктя)')
axes[0].set_xlabel('Number of clusters')
axes[0].set_ylabel('Inertia')
axes[0].grid(True)

axes[1].plot(n_range, silhouette, marker='o', color='green')
axes[1].set_title('Silhouette Score (більше - краще)')
axes[1].set_xlabel('Number of clusters')
axes[1].set_ylabel('Score')
axes[1].grid(True)

axes[2].plot(n_range, davies_bouldin, marker='o', color='red')
axes[2].set_title('Davies-Bouldin Index (менше - краще)')
axes[2].set_xlabel('Number of clusters')
axes[2].set_ylabel('Score')
axes[2].grid(True)

axes[3].plot(n_range, calinski_harabasz, marker='o', color='purple')
axes[3].set_title('Calinski-Harabasz Score (більше - краще)')
axes[3].set_xlabel('Number of clusters')
axes[3].set_ylabel('Score')
axes[3].grid(True)

plt.tight_layout()
plt.show()

#===========================================================

scalers = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
}

results = {
    scaler_name: {
        'inertia': [],
        'silhouette': [],
        'davies_bouldin': [],
        'calinski_harabasz': []
    }
    for scaler_name in scalers
}

for name, scaler in scalers.items():
    x_scaled = scaler.fit_transform(X)
    for n in n_range:
        clusterer = KMeans(n_clusters=n, n_init=10, random_state=52)
        clusterer.fit(x_scaled)
        labels = clusterer.labels_

        inertia = clusterer.inertia_
        silhouette = metrics.silhouette_score(X, labels)
        davies_bouldin = metrics.davies_bouldin_score(X, labels)
        calinski_harabasz = metrics.calinski_harabasz_score(X, labels)

        results[name]['inertia'].append(inertia)
        results[name]['silhouette'].append(silhouette)
        results[name]['davies_bouldin'].append(davies_bouldin)
        results[name]['calinski_harabasz'].append(calinski_harabasz)

fig, axes = plt.subplots(2, 2, figsize=(15, 9))
axes = axes.ravel()

metrics_names = ['inertia', 'silhouette', 'davies_bouldin', 'calinski_harabasz']
titles = [
    'Inertia (метод ліктя)',
    'Silhouette Score (більше - краще)',
    'Davies-Bouldin (менше - краще)',
    'Calinski-Harabasz (більше - краще)'
]

for i, metric_key in enumerate(metrics_names):
    ax = axes[i]

    for scaler_name, metrics_data in results.items():
        ax.plot(n_range, metrics_data[metric_key], marker='o', label=scaler_name)

    ax.set_title(titles[i])
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Score')
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()