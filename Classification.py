# This code is to implement classification methods.

# Importing all Libraries
import numpy as np
from sklearn.datasets import load_wine # Dataset that I will be using
from sklearn.tree import DecisionTreeClassifier, plot_tree # To implement DecisionTree Classifier
from sklearn.svm import SVC # To implement Support Vector Classifier
from sklearn.ensemble import RandomForestClassifier # To implement RandomForest Classifier
from sklearn.linear_model import LogisticRegression # To implement Logistic Regression Classifier
from sklearn.neighbors import KNeighborsClassifier # To implement K-Neighbors Classifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, roc_curve, \
    confusion_matrix, plot_roc_curve
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import pandas as pd # Storing data convenieniently
from sklearn.preprocessing import StandardScaler # To preprocessing
import matplotlib.pyplot as plt # Plotting
from sklearn import metrics

# Loading wine dataset
wine = load_wine()

X=wine.data    # assigning the features
Y=wine.target  # assigning the target variable

# Split the data into training (70%) and testing (30%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
print(X_train.shape)
print(X_test.shape)

# Preprocessing data
df_wine = pd.DataFrame(wine.data, columns=wine.feature_names) # Creating dataframe
std_wine = StandardScaler().fit_transform(df_wine) # normalizing the data

# Code reference link for 1st, 2nd, 3rd and 4th classification methods: https://github.com/Ayantika22/Kmeans-and-HCA-clustering-Visualization-for-WINE-dataset/blob/master/Mtech_Wine_kmeans%20and%20HCA.py
# Implementing 1st classification method (DecisionTreeClassifier)
plt.figure(figsize=[12,6])
print("\n1st Classification Method: DecisionTreeClassifier")
print("---------------------------------------------------")
clf = DecisionTreeClassifier().fit(wine.data,wine.target)
plot_tree(clf, filled=True, fontsize=7)
plt.title('Decision Tree of Wine Dataset', fontsize=26)
plt.show()

# K-fold cross-validation for K=10 for DecisionTreeClassifier
decision_tree = DecisionTreeClassifier(max_depth=5, random_state=0)
kfold = KFold(n_splits=10, random_state=0, shuffle=True)
cross_val_decision = cross_val_score(decision_tree, X, Y, cv=kfold, scoring='accuracy')

clf_decision = decision_tree.fit(X_train,Y_train)
y_pred_decision = clf_decision.predict(X_test)
y_pred_decision_roc = clf_decision.predict_proba(X_test)

print("Accuracy Score:",accuracy_score(Y_test, y_pred_decision))
print("Balanced Accuracy: ", balanced_accuracy_score(Y_test, y_pred_decision))
print("F1-Score: ", f1_score(Y_test, y_pred_decision, average='micro'))
print("ROC AUC: ", roc_auc_score(Y_test, y_pred_decision_roc, multi_class='ovr'))
print("Confusion Matrix:\n",confusion_matrix(Y_test, y_pred_decision))

# Implementing 2nd classification method (SupportVectorClassifier)
print("\n2nd Classification Method: SupportVectorClassifier")
print("---------------------------------------------------")

# K-fold cross-validation for K=10 for SupportVectorClassifier
svc = SVC(kernel='rbf', random_state=0, probability=True)
kfold = KFold(n_splits=10, random_state=0, shuffle=True)
cross_val_svc = cross_val_score(svc, X, Y, cv=kfold, scoring='accuracy')

clf_svc = svc.fit(X_train,Y_train)
y_pred_svc = clf_svc.predict(X_test)
y_pred_svc_roc = clf_svc.predict_proba(X_test)

print("Accuracy Score:",accuracy_score(Y_test, y_pred_svc))
print("Balanced Accuracy: ", balanced_accuracy_score(Y_test, y_pred_svc))
print("F1-Score: ", f1_score(Y_test, y_pred_svc, average='micro'))
print("ROC AUC: ", roc_auc_score(Y_test, y_pred_svc_roc, multi_class='ovr'))
print("Confusion Matrix:\n",confusion_matrix(Y_test, y_pred_svc))

# Code reference link to plot SVC:https://github.com/kshitijved/Support_Vector_Machine/blob/master/SVM.txt
# To plot SVC
h = 0.02
x=wine.data[:,:2]# we are taking first two feature
y=wine.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
svc1 = SVC(kernel='rbf', C=1, random_state=0, probability=True).fit(x_train,y_train)
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
xx.shape

Z = svc1.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)

plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.title('SVC for Wine Dataset', fontsize=20)
plt.xlabel('alchohol', fontsize=15)
plt.ylabel('malic_acid', fontsize=15)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks()
plt.yticks()
plt.show()

# Implementing 3rd classification method (RandomForestClassifier)
print("\n3rd Classification Method: RandomForestClassifier")
print("---------------------------------------------------")

# K-fold cross-validation for K=10 for RandomForestClassifier
rand_forest = RandomForestClassifier(n_estimators=1, max_depth=5, random_state=0)
kfold = KFold(n_splits=10, random_state=0, shuffle=True)
cross_val_randf = cross_val_score(rand_forest, X, Y, cv=kfold, scoring='accuracy')

clf_rand_forest = rand_forest.fit(X_train,Y_train)
y_pred_rand_forest = clf_rand_forest.predict(X_test)
y_pred_rand_forest_roc = clf_rand_forest.predict_proba(X_test)

print("Accuracy Score:",accuracy_score(Y_test, y_pred_rand_forest))
print("Balanced Accuracy: ", balanced_accuracy_score(Y_test, y_pred_rand_forest))
print("F1-Score: ", f1_score(Y_test, y_pred_rand_forest, average='micro'))
print("ROC AUC: ", roc_auc_score(Y_test, y_pred_rand_forest_roc, multi_class='ovr'))
print("Confusion Matrix:\n",confusion_matrix(Y_test, y_pred_rand_forest))

# Implementing 4th classification method (Logistic regression)
print("\n4th Classification Method: Logistic regression")
print("---------------------------------------------------")

# K-fold cross-validation for K=10 for Logistic regression
log_reg = LogisticRegression(solver='saga',random_state = 0, max_iter = 10000)
kfold = KFold(n_splits=10, random_state=0, shuffle=True)
cross_val_log_reg = cross_val_score(log_reg, X, Y, cv=kfold, scoring='accuracy')

clf_log_reg = log_reg.fit(X_train,Y_train)
y_pred_log_reg = clf_log_reg.predict(X_test)
y_pred_log_reg_roc = clf_log_reg.predict_proba(X_test)

print("Accuracy Score:",accuracy_score(Y_test, y_pred_log_reg))
print("Balanced Accuracy: ", balanced_accuracy_score(Y_test, y_pred_log_reg))
print("F1-Score: ", f1_score(Y_test, y_pred_log_reg, average='micro'))
print("ROC AUC: ", roc_auc_score(Y_test, y_pred_log_reg_roc, multi_class='ovr'))
print("Confusion Matrix:\n",confusion_matrix(Y_test, y_pred_log_reg))

# Code Reference Link:  https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn
# Implementing 5th classification method (K-Neighbors Classification)
print("\n5th Classification Method: K-Neighbors Classifier")
print("---------------------------------------------------")

# K-fold cross-validation for K=10 for Logistic regression
knn = KNeighborsClassifier(n_neighbors=3)
kfold = KFold(n_splits=10, random_state=0, shuffle=True)
cross_val_log_reg = cross_val_score(knn, X, Y, cv=kfold, scoring='accuracy')

clf_knn = knn.fit(X_train,Y_train)
y_pred_knn = knn.predict(X_test)
y_pred_knn_roc = clf_knn.predict_proba(X_test)

print("Accuracy Score:",metrics.accuracy_score(Y_test,y_pred_knn))
print("Balanced Accuracy: ", balanced_accuracy_score(Y_test, y_pred_knn))
print("F1-Score: ", f1_score(Y_test, y_pred_knn, average='micro'))
print("ROC AUC: ", roc_auc_score(Y_test, y_pred_knn_roc, multi_class='ovo'))
print("Confusion Matrix:\n",confusion_matrix(Y_test, y_pred_knn))

# Code reference link for Roc curve: https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_roc_curve_visualization_api.html
# Implementing ROC classifier
Y=Y==2 # To convert into binary
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# Implementing ROC Curve for DecisionTreeClassifier
ax = plt.gca()
decision_roc_curve = decision_tree.fit(X_train,Y_train)
plot_roc_curve(decision_roc_curve, X_test, Y_test, ax = ax)

# Implementing ROC Curve for SupportVectorClassifier
svc_roc_curve = svc.fit(X_train,Y_train)
plot_roc_curve(svc_roc_curve, X_test, Y_test, ax=ax)

# Implementing ROC Curve for RandomForestClassifier
rand_forest_roc_curve = rand_forest.fit(X_train,Y_train)
plot_roc_curve(rand_forest_roc_curve, X_test, Y_test, ax = ax)

# Implementing ROC Curve for Logistic Regression
log_reg_roc_curve = log_reg.fit(X_train,Y_train)
plot_roc_curve(log_reg_roc_curve, X_test, Y_test, ax=ax)

# Implementing ROC Curve for K-Neighbors Classifier
knn_roc_curve = knn.fit(X_train,Y_train)
plot_roc_curve(knn_roc_curve, X_test, Y_test, ax=ax)
plt.title('ROC Curve', fontsize=26)
plt.show()

# Code reference link for comparing models: https://www.kaggle.com/abhikaggle8/wine-classification
# Results of all the algorithms
clf_models = []

clf_models.append(("Decision Tree Classifier: ", DecisionTreeClassifier(max_depth=5, random_state=0)))
clf_models.append(("Support Vector Machine: ", SVC(kernel="rbf", random_state=0, probability=True)))
clf_models.append(("Random Forest Classifier: ", RandomForestClassifier(n_estimators=1, max_depth=5, random_state=0)))
clf_models.append(("Logistic Regression: ", LogisticRegression(solver='saga',random_state=0, max_iter=10000)))
clf_models.append(("KNeighbors Classifier: ", KNeighborsClassifier(n_neighbors=3)))

print('\nClassification models results are...')
results = []
names = []
for name,model in clf_models:
    kfold = KFold(n_splits=10, random_state=0, shuffle=True)
    cv_result = cross_val_score(model,X,Y, cv = kfold, scoring = "accuracy")
    names.append(name)
    results.append(cv_result)
for i in range(len(names)):
    print(names[i],results[i].mean()*100)

# Configuration of the parameters
clf_models_chang_para = []

clf_models_chang_para.append(("Decision Tree Classifier: ", DecisionTreeClassifier(max_depth=5, random_state=90)))
clf_models_chang_para.append(("Support Vector Machine: ", SVC(kernel="linear", random_state=50)))
clf_models_chang_para.append(("Random Forest Classifier: ", RandomForestClassifier(n_estimators=20, max_depth=5, random_state=30)))
clf_models_chang_para.append(("Logistic Regression: ", LogisticRegression(solver='saga',random_state = 100, max_iter = 100000)))
clf_models_chang_para.append(("KNeighbors Classifier: ", KNeighborsClassifier(n_neighbors=1)))

print('\nClassification models results after changing parameters are...')
results_chang_para = []
names_chang_para = []
for name_chang_para,model_change_para in clf_models_chang_para:
    kfold_chang_para = KFold(n_splits=10, random_state=0, shuffle=True)
    cv_result_chang_para = cross_val_score(model_change_para,X,Y, cv = kfold_chang_para, scoring = "accuracy")
    names_chang_para.append(name_chang_para)
    results_chang_para.append(cv_result_chang_para)
for i in range(len(names_chang_para)):
    print(names_chang_para[i],results_chang_para[i].mean()*100)