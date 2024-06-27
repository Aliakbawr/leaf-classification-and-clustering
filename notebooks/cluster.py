import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score, homogeneity_score, completeness_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

file_path = '/content/drive/My Drive/programming/leaf-classification-clustering/data/leaves.csv'
df_clus = pd.read_csv(file_path, header=None)
column_names = [f'feature_{i}' for i in range(df_clus.shape[1])]
df_clus.columns = column_names

X = df_clus.drop(columns='feature_0')
y = df_clus['feature_0']

# Standardizing the features
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Clip outliers
X_scaled = np.clip(X_scaled, -3, 3)


# Applying PCA
pca = PCA(n_components=11)
X_pca = pca.fit_transform(X_scaled)

agglo = AgglomerativeClustering(n_clusters=30)
agglo_labels = agglo.fit_predict(X_pca)

# Evaluate Agglomerative Clustering
homogeneity_agglo = homogeneity_score(y, agglo_labels)
completeness_agglo = completeness_score(y, agglo_labels)
v_measure_agglo = v_measure_score(y, agglo_labels)
ari_agglo = adjusted_rand_score(y, agglo_labels)
nmi_agglo = normalized_mutual_info_score(y, agglo_labels)

print(f'Agglomerative Clustering Homogeneity: {homogeneity_agglo:.2f}')
print(f'Agglomerative Clustering Completeness: {completeness_agglo:.2f}')
print(f'Agglomerative Clustering V-measure: {v_measure_agglo:.2f}')
print(f'Agglomerative Clustering Adjusted Rand Index: {ari_agglo:.2f}')
print(f'Agglomerative Clustering Normalized Mutual Information: {nmi_agglo:.2f}')