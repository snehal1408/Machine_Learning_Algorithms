# In this file, We are loading the wine dataset using SciKit
# And then, we will identify its key aspects (number of dimensions/features, number and names of classes, number of samples per class, etc.).
import numpy as np
from scipy.stats import stats
from sklearn.datasets import load_wine
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

wine = load_wine() # This will load wine dataset and store in the variable called wine
n_samples, n_features = wine.data.shape # This will define the samples, features and store them in n_samples, n_feature variable respectively.
df_wine = pd.DataFrame(wine.data, columns=wine.feature_names) # Create dataframe for wine dataset

# Code Reference Link: https://towardsdatascience.com/comparing-classification-models-for-wine-quality-prediction-6c5f26669a4f
# Printing first 4 rows of wine dataset
wine_head = df_wine.head(4)
print("First five rows of Wine dataset:")
print("-----------------------------------------------------------")
print(wine_head.to_string()) # To see whole columns

# Printing last 4 rows of wine dataset
wine_tail = df_wine.tail(4)
print("\nLast five rows of Wine dataset:")
print("-----------------------------------------------------------")
print(wine_tail.to_string()) # To see whole columns

# Summary of the dataset
print("\nSummary of Wine dataset:")
print("-----------------------------------------------------------")
print(df_wine.describe().to_string()) # To see whole columns

# Checking the Null values in wine dataset
print("\nChecking null values in Wine dataset:")
print("-----------------------------------------------------------")
wine_null = df_wine.isnull().sum()
print(str(wine_null))

# To show Outlier
z = np.abs(stats.zscore(df_wine))
df_wine =df_wine[(z < 3).all(axis=1)]
var = df_wine.shape
print(var)
sns.boxplot(df_wine['color_intensity'])

# Showing relationship between Different variables in Wine Dataset using HeatMap
corr = df_wine[df_wine.columns].corr()
plt.subplots(figsize=(12,7))
b=sns.heatmap(corr, annot = True, cbar=False)
plt.xticks(rotation=23,fontsize=7)
plt.yticks(rotation=30,fontsize=7)
plt.show()

print("\nNumber of Dimensions/Features: " + str(n_features)) # This will print the number of the Dimensions/Features of the dataset
print("Name of Dimensions/Features: " + str(wine.feature_names) ) # This will print the names of the Dimensions/Features of the dataset
print("Number of classes: " + str(len(wine.target_names)) + "\nNames of classes: " + str(wine.target_names)) # This will print the number and names of the classes in the datset
print("Total number of Samples: " + str(n_samples)) # This will print Total number of Samples in the dataset
print("Number of Samples per each class: " + str(np.bincount(wine.target))) # This will print number of Samples per each class in the dataset
