# This code is to implement Clustering methods.

# Importing all Libraries
from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift, Birch  # Our clustering algorithm
from sklearn.mixture import GaussianMixture # To implement GaussianMixture model clustering
from sklearn.decomposition import PCA # Needed for dimension reduction
from sklearn.datasets import load_wine # Dataset that I will be using
import matplotlib.pyplot as plt # Plotting
import pandas as pd # Storing data convenieniently
from sklearn.metrics import adjusted_rand_score, homogeneity_score, v_measure_score, completeness_score, \
    adjusted_mutual_info_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler # To preprocessing
import scipy.cluster.hierarchy as sch
from sklearn import metrics

# Loading wine dataset
wine = load_wine()

X = wine.data    # assigning the features
Y = wine.target  # assigning the target variable

# Split the data into training (70%) and testing (30%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
print(X_train.shape)
print(X_test.shape)
labels_true = Y

# Code reference link for finding score in all clustering methods: https://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/modules/clustering.html
# Code reference link for K-means with PCA: https://andrewmourcos.github.io/blog/2019/06/06/PCA.html
# Code reference link for showing iteration with k-means: https://sweetcode.io/k-means-clustering-python/

# Preprocessing data
df_wine = pd.DataFrame(wine.data, columns=wine.feature_names) # Creating dataframe
std_wine = StandardScaler().fit_transform(df_wine) # normalizing the data

# 1st Clustering Method (K-means)
print("\n1st Clustering Method: K-means")
print("---------------------------------------------------")

model_kmeans = KMeans(n_clusters=3)# Build the model
model_kmeans.fit(X)# Fit the model on the dataset
labels_kmeans = model_kmeans.labels_

print("Adjusted Random Score: ", adjusted_rand_score(labels_true, labels_kmeans))
print("Homogeneity Score: ", homogeneity_score(labels_true, labels_kmeans))
print("V-Measure Score: ", v_measure_score(labels_true, labels_kmeans))
print("Completeness: ", completeness_score(labels_true, labels_kmeans))
print("Adjusted Mutual Information: ", adjusted_mutual_info_score(labels_true, labels_kmeans))

plt.scatter(X[:,0],X[:,12], c=labels_kmeans, cmap='plasma')
plt.title('Kmeans Clustering for Wine Dataset', fontsize=20)
plt.xlabel('alchohol', fontsize=15)
plt.ylabel('proline', fontsize=15)
plt.show()

# Kmeans with Dimentionality Reduction using PCA
pca_wine = PCA(n_components=13)
principalComponents = pca_wine.fit_transform(std_wine)
PCA_components = pd.DataFrame(principalComponents) # Putting components in a dataframe for later

array_pca=[]
for i in range(1, 5):
    model_kmeans_pca = KMeans(n_clusters = i, init = 'k-means++', max_iter = 1, n_init = 10, random_state = 5)
    model_kmeans_pca.fit(PCA_components.iloc[:, :2])
    labels = model_kmeans_pca.predict(PCA_components.iloc[:, :2])
    centroids = model_kmeans_pca.cluster_centers_
    array_pca.append(model_kmeans_pca.inertia_)
    plt.scatter(PCA_components[0], PCA_components[1], c=labels)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=200, marker='*',color='r')
    plt.title("Wine Dataset with " + str(i) + " Clusters", fontsize=20)
    plt.show()

# Comparing 'alcohol' and 'od280/od315_of_diluted_wines' of wine dataset with Kmeans for showing iteration
df_wine.plot.scatter(x = 'alcohol', y = 'od280/od315_of_diluted_wines', figsize=(12,8), colormap='jet') # Plot scatter plot of proline and proline
plt.title('Wine Dataset with no Iteration performed', fontsize=26)
plt.show()

# 1st iteration
model_kmeans_iter = KMeans(n_clusters=3, init = 'random', max_iter = 300, random_state = 5).fit(df_wine.iloc[:,[11,0]])
centroids_df = pd.DataFrame(model_kmeans_iter.cluster_centers_, columns = list(df_wine.iloc[:,[11,0]].columns.values))
fig, ax = plt.subplots(1, 1)
df_wine.plot.scatter(x = 'alcohol', y = 'od280/od315_of_diluted_wines', c= model_kmeans_iter.labels_, figsize=(12,8), colormap='jet', ax=ax, mark_right=False)
centroids_df.plot.scatter(x = 'alcohol', y = 'od280/od315_of_diluted_wines', ax = ax,  s = 200, marker='*', c='m', mark_right=False)
plt.title('1st Iteration performed on Wine Dataset', fontsize=26)
plt.show()

# 2nd iteration
model_kmeans_iter = KMeans(n_clusters=3, init = 'random', max_iter = 1, random_state = 5).fit(df_wine.iloc[:,[11,0]])
centroids_df = pd.DataFrame(model_kmeans_iter.cluster_centers_, columns = list(df_wine.iloc[:,[11,0]].columns.values))
fig, ax = plt.subplots(1, 1)
df_wine.plot.scatter(x = 'alcohol', y = 'od280/od315_of_diluted_wines', c= model_kmeans_iter.labels_, figsize=(12,8), colormap='jet', ax=ax, mark_right=False)
centroids_df.plot.scatter(x = 'alcohol', y = 'od280/od315_of_diluted_wines', ax = ax,  s = 200, marker='*', c='m',mark_right=False)
plt.title('2nd Iteration performed on Wine Dataset', fontsize=26)
plt.show()

# Count accuracy after last iteration
X_train_iter, X_test_iter, Y_train_iter, Y_test_iter = train_test_split(df_wine.iloc[:,[11,0]], Y, test_size=0.3, random_state=0)
y_pred_iter = model_kmeans_iter.predict(X_test_iter)
model_kmeans_iter_acc = metrics.accuracy_score(Y_test_iter,y_pred_iter)
print('Accuracy after iteration performed: {0:f}'.format(model_kmeans_iter_acc))

# Code reference link for Hierarchical dendogram and agglomerative clustering: https://github.com/Ayantika22/Kmeans-and-HCA-clustering-Visualization-for-WINE-dataset/blob/master/Mtech_Wine_kmeans%20and%20HCA.py
# 2nd Clustering Method (Hierarchical Clustering & Agglomerative Hierarchical clustering)

# Plotting Hierarchical Dendrogram for Wine dataset
# Decide the number of clusters by using this dendrogram

model_hcl = sch.linkage(X, method = 'median')
plt.figure(figsize=(12,8))
den = sch.dendrogram(model_hcl)
plt.title('Dendrogram for the Clustering of the Wine Dataset', fontsize=26)
plt.xlabel('Type', fontsize=18)
plt.ylabel('Euclidean distance in the space with other variables', fontsize=18)
plt.show()

# Hierarchical Agglomerative Clustering
print("\n2nd Clustering Method: Hierarchical Agglomerative")
print("---------------------------------------------------")

model_agg = AgglomerativeClustering(n_clusters=3) # Build the Model
model_agg.fit(X) # Fit the model on the dataset
labels_agg = model_agg.labels_

print("Adjusted Random Score: ", adjusted_rand_score(labels_true, labels_agg))
print("Homogeneity Score: ", homogeneity_score(labels_true, labels_agg))
print("V-Measure Score: ", v_measure_score(labels_true, labels_agg))
print ("Completeness: ", completeness_score(labels_true, labels_agg))
print ("Adjusted Mutual Information: ", adjusted_mutual_info_score(labels_true, labels_agg))

plt.scatter(X[:,0],X[:,12], c=labels_agg, cmap='rainbow')
plt.title('Agglomerative Clustering for Wine Dataset', fontsize=20)
plt.xlabel('alchohol', fontsize=15)
plt.ylabel('proline', fontsize=15)
plt.show()

# Code reference link for GausianMixtureModel: https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
# 3rd Clustering Method (GausianMixtureModel Clustering)
print("\n3rd Clustering Method: GausianMixtureModel")
print("---------------------------------------------------")

model_gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)# Build the model
model_gmm.fit(X)# Fit the model on the dataset
labels_gmm = model_gmm.predict(X)

print("Adjusted Random Score: ", adjusted_rand_score(labels_true, labels_gmm))
print("Homogeneity Score: ", homogeneity_score(labels_true, labels_gmm))
print("V-Measure Score: ", v_measure_score(labels_true, labels_gmm))
print("Completeness: ", completeness_score(labels_true, labels_gmm))
print("Adjusted Mutual Information: ", adjusted_mutual_info_score(labels_true, labels_gmm))

plt.scatter(X[:, 0], X[:, 12], c=labels_gmm)
plt.title('GausianMixtureModel for Wine Dataset', fontsize=20)
plt.xlabel('alchohol', fontsize=15)
plt.ylabel('proline', fontsize=15)
plt.show()

# Code reference link for MeanShift: https://www.geeksforgeeks.org/ml-mean-shift-clustering/
# 4th Clustering Method (MeanShift Clustering)
print("\n4th Clustering Method: MeanShift Clustering")
print("---------------------------------------------------")

model_ms = MeanShift()# Build the model
model_ms.fit(X)# Fit the model on the dataset
cluster_centers = model_ms.cluster_centers_
labels_ms = model_ms.labels_

print("Adjusted Random Score: ", adjusted_rand_score(labels_true, labels_ms))
print("Homogeneity Score: ", homogeneity_score(labels_true, labels_ms))
print("V-Measure Score: ", v_measure_score(labels_true, labels_ms))
print ("Completeness: ", completeness_score(labels_true, labels_ms))
print ("Adjusted Mutual Information: ", adjusted_mutual_info_score(labels_true, labels_ms))

plt.scatter(X[:,0], X[:,12],c=labels_ms, marker='o')
plt.scatter(cluster_centers[:,0], cluster_centers[:,12], marker='x', color='red', linewidth=2)
plt.title('MeanShift Clustering for Wine Dataset', fontsize=20)
plt.xlabel('alchohol', fontsize=15)
plt.ylabel('proline', fontsize=15)
plt.show()

# Code reference link for Birch Clustering: https://www.kaggle.com/ninjacoding/clustering-mall-customers#Birch
# 5th Clustering Method (Birch Clustering)
print("\n5th Clustering Method: Birch Clustering")
print("---------------------------------------------------")
model_birch=Birch(n_clusters=3)# Build the model
model_birch.fit(X)# Fit the model on the dataset
labels_birch = model_birch.labels_

print("Adjusted Random Score: ", adjusted_rand_score(labels_true, labels_birch))
print("Homogeneity Score: ", homogeneity_score(labels_true, labels_birch))
print("V-Measure Score: ", v_measure_score(labels_true, labels_birch))
print ("Completeness: ", completeness_score(labels_true, labels_birch))
print ("Adjusted Mutual Information: ", adjusted_mutual_info_score(labels_true, labels_birch))

plt.scatter(X[:,0],X[:,12],c=labels_birch,cmap='coolwarm')
plt.title('Birch Clustering for Wine Dataset', fontsize=20)
plt.xlabel('alchohol', fontsize=15)
plt.ylabel('proline', fontsize=15)
plt.show()

# Results of all the algorithms
clu_models = []

clu_models.append(("Kmeans Clustering: ", KMeans(n_clusters=1)))
clu_models.append(("Gaussian Mixture Model: ", GaussianMixture(n_components=2, covariance_type='full', random_state=0)))
clu_models.append(("MeanShift Clustering: ", MeanShift()))
clu_models.append(("Birch Clustering:",Birch(n_clusters=1)))

print('\nClustering models results are...')
results = []
names = []
for name, model in clu_models:
    kfold = KFold(n_splits=10, random_state=0, shuffle=True)
    cv_result_chang_para = cross_val_score(model,X,Y, cv = kfold, scoring = "accuracy")
    names.append(name)
    results.append(cv_result_chang_para)
for i in range(len(names)):
    print(names[i],results[i].mean()*100)
model_agg = AgglomerativeClustering(n_clusters=1,linkage='complete')
model_agg.fit(X)
labels_agg = model_agg.labels_
print("Agglomerative Clustering: ", accuracy_score(labels_true, labels_agg).mean()*100)

# Configuration of the parameters
clf_models_chang_para = []

clf_models_chang_para.append(("Kmeans Clustering: ", KMeans(n_clusters=1, max_iter=20, random_state=20, init='random')))
clf_models_chang_para.append(("Gaussian Mixture Model: ", GaussianMixture(n_components=2, covariance_type='tied', random_state=0)))
clf_models_chang_para.append(("MeanShift Clustering: ", MeanShift(max_iter=1)))
clf_models_chang_para.append(("Birch Clustering:", Birch(n_clusters=3)))

print('\nClustering models results after changing parameters are...')
results_chang_para = []
names_chang_para = []
for name_chang_para,model_change_para in clf_models_chang_para:
    kfold_chang_para = KFold(n_splits=10, random_state=0, shuffle=True)
    cv_result_chang_para = cross_val_score(model_change_para,X,Y, cv = kfold_chang_para, scoring = "accuracy")
    names_chang_para.append(name_chang_para)
    results_chang_para.append(cv_result_chang_para)
for i in range(len(names_chang_para)):
    print(names_chang_para[i],results_chang_para[i].mean()*100)
model_agg = AgglomerativeClustering(n_clusters=3,linkage='ward')
model_agg.fit(X)
labels_agg = model_agg.labels_
print("Agglomerative Clustering: ", accuracy_score(labels_true, labels_agg).mean()*100)











