import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from IPython.display import clear_output
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns

path = '/Users/willweiwu/Library/CloudStorage/Dropbox/CU Boulder/2025 Spring/INFO 5653-001 Text Mining/Project/wiki/vectorized_data_wiki.csv'
DF = pd.read_csv(path)
OriginalDF = DF.copy()

DFLabel = DF["sentiment"]
print(DFLabel)
DFLabel_string = DFLabel

DF = DF.drop(["sentiment"], axis=1)
print(DF)

### Standardize  dataset
scaler = StandardScaler()
DF = scaler.fit_transform(DF)
print(DF)
print(type(DF))
print(DF.shape)

### PERFORM PCA

MyPCA = PCA(n_components=3)
Result = MyPCA.fit_transform(DF)
print(Result)
print("The eigenvalues:", MyPCA.explained_variance_)
MyCov = np.cov(Result.T)
print("Covar of the PC PCA Matrix: \n", MyCov)
print("The relative eigenvalues are:", MyPCA.explained_variance_ratio_)
print("The actual eigenvalues are:", MyPCA.explained_variance_)
EVects = MyPCA.components_
print("The eigenvectors are:\n", EVects)


### K MEANS CLUSTERING

kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(Result)
print("Cluster Labels:", labels)


### 3D Visualization

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Result[:, 0], Result[:, 1], Result[:, 2], c=labels, cmap='viridis', s=50)
ax.set_title('3D PCA with Clustering')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.show()


### Top words in each PC

feature_names = OriginalDF.columns[1:]

top_n = 10

for i in range(MyPCA.n_components):
    print(f"\nTop {top_n} features for Principal Component {i+1}:")
    
    component = EVects[i]
    
    feature_loadings = pd.DataFrame({
        'Feature': feature_names,
        'Loading': component
    })
    
    feature_loadings['AbsLoading'] = feature_loadings['Loading'].abs()
    top_features = feature_loadings.sort_values(by='AbsLoading', ascending=False).head(top_n)
    
    print(top_features[['Feature', 'Loading']])


