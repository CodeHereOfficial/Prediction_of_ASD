#!/usr/bin/env python
# coding: utf-8

# ## Road Map:
# 
# * [Step 0](#step0): Import Datasets.
# 
# * [Step 1](#step1): Clean Datasets (The data needs to be cleaned; many rows contain missing data, and there may be erroneous data identifiable as outliers).
# 
# * [Step 2](#step2): A quick visualization with *Seaborn*.
# 
# * [Step 3](#step3): At First, We applied several Supervised Machine Learning (SML) techniques on the data for classification purpose.
# 
# * [Step 4](#step4): Next, We experimented with different topologies, optimizers, and hyperparameters for different models.
# 
# * [Step 5](#step5):Test the Modals
# 
# 
# 
# 

# # Step 0: Import Datasets
#     We start by importing the 'ASD_data.csv' file into a Pandas dataframe and take a look at it.

# In[141]:


# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames


# Pretty display for notebooks
get_ipython().run_line_magic('matplotlib', 'inline')


data = pd.read_csv('ASD_data.csv')

data.head(n=5)


# In[142]:


# Total number of records
n_records = len(data.index)

# TODO: Number of records where individual's with ASD
n_asd_yes = len(data[data['Class'] == 'YES'])

# TODO: Number of records where individual's with no ASD
n_asd_no = len(data[data['Class'] == 'NO'])

# TODO: Percentage of individuals whose are with ASD
yes_percent = float(n_asd_yes) / n_records *100

# Print the results
print ("Total number of records: {}".format(n_records))
print ("Individuals diagonised with ASD: {}".format(n_asd_yes))
print ("Individuals not diagonised with ASD: {}".format(n_asd_no))
print ("Percentage of individuals diagonised with ASD: {:.2f}%".format(yes_percent))


# **Preparing the Data**
# Before data can be used as input for machine learning algorithms, it must be cleaned, formatted, and maybe even restructured — this is typically known as preprocessing. Unfortunately, for this dataset, there are many invalid or missing entries(?) we must deal with, moreover, there are some qualities about certain features that must be adjusted. This preprocessing can help tremendously with the outcome and predictive power of nearly all learning algorithms.
# 
# We use the optional parmaters in read_csv to convert missing data (indicated by a ?) into NaN, and to add the appropriate column names ():

# In[143]:


asd_data = pd.read_csv('ASD_data.csv', na_values=['?']) #replacing '?' with NaN.
asd_data.head(n=5)


# In[144]:


asd_data.describe()


# # Step 1: Clean Datasets

# In[145]:


asd_data.loc[(asd_data['age'].isnull()) |(asd_data['gender'].isnull()) |(asd_data['ethnicity'].isnull()) 
|(asd_data['jundice'].isnull())|(asd_data['austim'].isnull()) |(asd_data['contry_of_res'].isnull())
            |(asd_data['used_app_before'].isnull())|(asd_data['result'].isnull())|(asd_data['age_desc'].isnull())
            |(asd_data['relation'].isnull())]  #This selects the rows with missing data.


# In[146]:


asd_data.dropna(inplace=True)  #delete rows with missing data. 
asd_data.describe()


# Let's check out the data types of all our features including the target feature. Moreover, lets count the total number of instances and the target-class distribution.

# In[147]:


# Reminder of the features:
print(asd_data.dtypes)


# Total number of records in clean dataset
n_records = len(asd_data.index)

# TODO: Number of records where individual's with ASD in the clean dataset
n_asd_yes = len(asd_data[asd_data['Class'] == 'YES'])

# TODO: Number of records where individual's with no ASD in the clean dataset
n_asd_no = len(asd_data[asd_data['Class'] == 'NO'])

# Print the results
print ("Total number of records: {}".format(n_records))
print ("Individuals diagonised with ASD: {}".format(n_asd_yes))
print ("Individuals not diagonised with ASD: {}".format(n_asd_no))


# # Step 2: A quick visualization with Seaborn

# In[148]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid", color_codes=True)


# In[149]:


# Draw a nested violinplot and split the violins for easier comparison #hue- takes column name for color encoding.
sns.violinplot(x="jundice", y="result", hue="austim", data=asd_data, split=True,
                inner="box", palette={'yes': "r", 'no': "b"})


# In[150]:


# Draw a nested violinplot and split the violins for easier comparison
sns.violinplot(x="jundice", y="result", hue="Class", data=asd_data, split=True,
                inner="quart", palette={'YES': "r", 'NO': "b"})


# In[151]:


# Draw a nested violinplot and split the violins for easier comparison
sns.violinplot(x="gender", y="result", hue="Class", data=asd_data, split=True,
                inner="quart", palette={'YES': "r", 'NO': "b"})
#sns.despine(left=True)


# In[152]:


sns.factorplot(x="jundice", y="result", hue="Class", col="gender", data=asd_data, kind="swarm");


# In[153]:


sns.factorplot(x="gender", y="result", hue="Class",
               col="relation", data=asd_data, kind="box", size=4, aspect=.5, palette={'YES': "r", 'NO': "b"});


# In[162]:


#multiploting
g = sns.factorplot(x="result", y="jundice",
                   hue="gender", row="relation",
                   data=asd_data,
                    orient="h", size=2, aspect=3.5, palette={'f': "r", 'm': "b"},
                  kind="violin", dodge=True, cut=0, bw=.2)


# In[163]:


asd_data.tail()


# In[164]:


asd_data.Class .unique()


# In[165]:


asd_data['Class'].replace('NO', 0, inplace=True)
asd_data['Class'].replace('YES', 1, inplace=True)


# In[166]:


asd_data.head()


# In[167]:


asd_data.relation.unique()


# In[168]:



asd_data['relation'].replace('Health care professional', 0, inplace=True)
asd_data['relation'].replace('Others', 1, inplace=True)
asd_data['relation'].replace('Parent', 2, inplace=True)
asd_data['relation'].replace('Relative', 3, inplace=True)
asd_data['relation'].replace('Self', 4, inplace=True)


# In[169]:


asd_data.age_desc.unique()


# In[170]:


asd_data['age_desc'].replace('18 and more', 1, inplace=True)
asd_data['age_desc'].replace('Less than 18', 0, inplace=True)


# In[171]:


asd_data.used_app_before.unique()


# In[172]:


asd_data['used_app_before'].replace('no', 0, inplace=True)
asd_data['used_app_before'].replace('yes', 1, inplace=True)


# In[173]:


asd_data.contry_of_res.unique()


# In[174]:


asd_data['contry_of_res'].replace('United States', 0, inplace=True)
asd_data['contry_of_res'].replace('Brazil', 1, inplace=True)
asd_data['contry_of_res'].replace('Spain', 2, inplace=True)
asd_data['contry_of_res'].replace('New Zealand', 3, inplace=True)
asd_data['contry_of_res'].replace('Bahamas', 4, inplace=True)
asd_data['contry_of_res'].replace('Burundi', 5, inplace=True)
asd_data['contry_of_res'].replace('Jordan', 6, inplace=True)
asd_data['contry_of_res'].replace('Ireland', 7, inplace=True)
asd_data['contry_of_res'].replace('United Arab Emirates', 8, inplace=True)
asd_data['contry_of_res'].replace('Afghanistan', 9, inplace=True)
asd_data['contry_of_res'].replace('United Kingdom', 10, inplace=True)
asd_data['contry_of_res'].replace('South Africa', 11, inplace=True)
asd_data['contry_of_res'].replace('Italy', 12, inplace=True)
asd_data['contry_of_res'].replace('Pakistan', 13, inplace=True)
asd_data['contry_of_res'].replace('Egypt', 14, inplace=True)
asd_data['contry_of_res'].replace('Bangladesh', 15, inplace=True)
asd_data['contry_of_res'].replace('Chile', 16, inplace=True)
asd_data['contry_of_res'].replace('France', 17, inplace=True)
asd_data['contry_of_res'].replace('China', 18, inplace=True)
asd_data['contry_of_res'].replace('Australia', 19, inplace=True)
asd_data['contry_of_res'].replace('Canada', 20, inplace=True)
asd_data['contry_of_res'].replace('Saudi Arabia', 21, inplace=True)
asd_data['contry_of_res'].replace('Netherlands', 22, inplace=True)
asd_data['contry_of_res'].replace('Romania', 23, inplace=True)
asd_data['contry_of_res'].replace('Sweden', 24, inplace=True)
asd_data['contry_of_res'].replace('Tonga', 25, inplace=True)
asd_data['contry_of_res'].replace('Oman', 26, inplace=True)
asd_data['contry_of_res'].replace('India', 27, inplace=True)
asd_data['contry_of_res'].replace('Philippines', 28, inplace=True)
asd_data['contry_of_res'].replace('Sri Lanka', 29, inplace=True)
asd_data['contry_of_res'].replace('Sierra Leone', 30, inplace=True)
asd_data['contry_of_res'].replace('Ethiopia', 31, inplace=True)
asd_data['contry_of_res'].replace('Viet Nam', 32, inplace=True)
asd_data['contry_of_res'].replace('Iran', 33, inplace=True)
asd_data['contry_of_res'].replace('Costa Rica', 34, inplace=True)
asd_data['contry_of_res'].replace('Germany', 35, inplace=True)
asd_data['contry_of_res'].replace('Mexico', 36, inplace=True)
asd_data['contry_of_res'].replace('Armenia', 37, inplace=True)
asd_data['contry_of_res'].replace('Iceland', 38, inplace=True)
asd_data['contry_of_res'].replace('Nicaragua', 39, inplace=True)
asd_data['contry_of_res'].replace('Austria', 40, inplace=True)
asd_data['contry_of_res'].replace('Russia', 41, inplace=True)
asd_data['contry_of_res'].replace('AmericanSamoa', 42, inplace=True)
asd_data['contry_of_res'].replace('Uruguay', 43, inplace=True)
asd_data['contry_of_res'].replace('Ukraine', 44, inplace=True)
asd_data['contry_of_res'].replace('Serbia', 45, inplace=True)
asd_data['contry_of_res'].replace('Portugal', 46, inplace=True)
asd_data['contry_of_res'].replace('Malaysia', 47, inplace=True)
asd_data['contry_of_res'].replace('Ecuador', 48, inplace=True)
asd_data['contry_of_res'].replace('Niger', 49, inplace=True)
asd_data['contry_of_res'].replace('Belgium', 50, inplace=True)
asd_data['contry_of_res'].replace('Bolivia', 51, inplace=True)
asd_data['contry_of_res'].replace('Aruba', 52, inplace=True)
asd_data['contry_of_res'].replace('Finland', 53, inplace=True)
asd_data['contry_of_res'].replace('Turkey', 54, inplace=True)
asd_data['contry_of_res'].replace('Nepal', 55, inplace=True)
asd_data['contry_of_res'].replace('Indonesia', 56, inplace=True)
asd_data['contry_of_res'].replace('Angola', 57, inplace=True)
asd_data['contry_of_res'].replace('Czech Republic', 58, inplace=True)
asd_data['contry_of_res'].replace('Cyprus', 59, inplace=True)


# In[175]:


asd_data.austim.unique()


# In[176]:


asd_data['austim'].replace('no', 0, inplace=True)
asd_data['austim'].replace('yes', 1, inplace=True)


# In[177]:


asd_data.jundice.unique()


# In[178]:


asd_data['jundice'].replace('no', 0, inplace=True)
asd_data['jundice'].replace('yes', 1, inplace=True)


# In[179]:


asd_data.ethnicity.unique()


# In[180]:


asd_data['ethnicity'].replace('White-European', 0, inplace=True)
asd_data['ethnicity'].replace('Latino', 1, inplace=True)
asd_data['ethnicity'].replace('Others', 2, inplace=True)
asd_data['ethnicity'].replace('Black', 3, inplace=True)
asd_data['ethnicity'].replace('Asian', 4, inplace=True)
asd_data['ethnicity'].replace('Middle Eastern ', 5, inplace=True)
asd_data['ethnicity'].replace('Pasifika', 6, inplace=True)
asd_data['ethnicity'].replace('South Asian', 7, inplace=True)
asd_data['ethnicity'].replace('Hispanic', 8, inplace=True)
asd_data['ethnicity'].replace('Turkish', 9, inplace=True)
asd_data['ethnicity'].replace('others', 2, inplace=True)


# In[181]:


asd_data.gender.unique()


# In[182]:


asd_data['gender'].replace('f', 0, inplace=True)
asd_data['gender'].replace('m', 1, inplace=True)


# In[183]:


asd_data.head()


# In[184]:


# Split the data into features and target label, converting dataframe into arrays to wrk with scikit_liearn.
asd_class = asd_data['Class']
features_raw = asd_data[['age', 'gender', 'ethnicity', 'jundice', 'austim', 'contry_of_res', 'result',
                      'relation','A1','A2','A3','A4','A5','A6','A7','A8',
                      'A9','A10']]


# In[185]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
numerical = ['age', 'result', 'contry_of_res', 'ethnicity' ]

features_final = pd.DataFrame(data = features_raw)
features_final[numerical] = scaler.fit_transform(features_raw[numerical])
features_final
# Show an example of a record with scaling applied
display(features_final.head(n = 5))


# In[186]:


features_final.isnull().mean()


# **Shuffle and Split Data**
# 
# Now all categorical variables have been converted into numerical features, and all numerical features have been normalized. As always, We will now split the data (both features and their labels) into training and test sets. 80% of the data will be used for training and 20% for testing.

# In[187]:


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(features_final, asd_class, train_size=0.80, random_state=1)


# Show the results of the split
print ("Training set has {} samples.".format(X_train.shape[0]))
print ("Testing set has {} samples.".format(X_test.shape[0]))
#asd_data


# # Step 3: Models
# **Supervised Learning Models**
# We have applied the following supervised learning models in this project which are currently available in scikit-learn.
# 
# (1) Decision Trees
# 
# (2) Random Forest
# 
# (3) Support Vector Machines (SVM)
# 
# (4) K-Nearest Neighbors (KNeighbors)
# 
# (5) Gaussian Naive Bayes (GaussianNB)
# 
# (6) Logistic Regression (LR)
# 
# (7) Linear Discriminant Analysis (LDA)
# 
# (8) Quadratic Discriminant Analysis (QDA)
#  
# 

# <a id='2'></a>
# ## (1) Decision Trees
# We start with creating a DecisionTreeClassifier and fit it to the training data.

# In[39]:


from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

dectree = DecisionTreeClassifier(random_state=1)  #crate a DT and save it as dectree

# Train the classifier on the training set
dectree.fit(X_train, y_train)


# In[40]:


dectree = dectree.fit(X_train, y_train)


# <a id='2'></a>
# ## Predictions and Evaluation of DT Model

# In[41]:


predictions_dectree = dectree.predict(X_test)
acc_dectree = accuracy_score(y_true = y_test, y_pred = predictions_dectree )
print("Overall accuracy of DT using test-set is : %f" %(acc_dectree*100))


# In[42]:


print(classification_report(y_test, predictions_dectree))


# In[43]:


print(confusion_matrix(y_test, predictions_dectree))


# **Depiction of Decision Tree algorithm**

# In[44]:



import pydotplus 
import graphviz

from IPython.display import Image
from sklearn.tree import export_graphviz
from six import StringIO
import pydotplus

#create DOT data
dot_data = tree.export_graphviz(dectree,
                                out_file=None,
                                filled=True,
                                rounded=True,
                                special_characters=True)  

#create graph
graph = pydotplus.graph_from_dot_data(dot_data)  

from IPython.display import Image 

#show graph
Image(graph.create_png()) 
 


# **Evaluating Model Performance**

# **Metrics**
# 
# We can use F-beta score as a metric that considers both precision and recall:
# 
# 𝐹𝛽=(1+𝛽2)⋅(𝑝𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛⋅𝑟𝑒𝑐𝑎𝑙𝑙/(𝛽2⋅𝑝𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛)+𝑟𝑒𝑐𝑎𝑙𝑙)
#  
# In particular, when  𝛽=0.5 , more emphasis is placed on precision. This is called the F 0.5  score (or F-score for simplicity).
# 
# **Note** : Recap of accuracy, precision, recall
# 
# **Accuracy** measures how often the classifier makes the correct prediction. It’s the ratio of the number of correct predictions to the total number of predictions (the number of test data points).
# 
# **Precision** tells us what proportion of messages we classified as spam, actually were spam. It is a ratio of true positives(words classified as spam, and which are actually spam) to all positives(all words classified as spam, irrespective of whether that was the correct classificatio), in other words it is the ratio of
# 
# [True Positives/(True Positives + False Positives)]
# 
# **Recall (sensitivity)** tells us what proportion of messages that actually were spam were classified by us as spam. It is a ratio of true positives(words classified as spam, and which are actually spam) to all the words that were actually spam, in other words it is the ratio of
# 
# [True Positives/(True Positives + False Negatives)]

# In[45]:


# make class predictions for the testing set
y_pred_class = dectree.predict(X_test)


# In[46]:


# print the first 25 true and predicted responses
print('True:', y_test.values[0:25])
print('False:', y_pred_class[0:25])


# In[47]:


from sklearn import metrics
# IMPORTANT: first argument is true values, second argument is predicted values
# this produces a 2x2 numpy array (matrix)
#print(metrics.confusion_matrix(y_test, y_pred_class))

# save confusion matrix and slice into four pieces
confusion = metrics.confusion_matrix(y_test, y_pred_class)
print(confusion)
#[row, column]
TP = confusion[1, 1]    #43
TN = confusion[0, 0]    #79
FP = confusion[0, 1]    #0
FN = confusion[1, 0]    #0


# **Metrics computed from a confusion matrix**

# **Classification Accuracy** : Overall, how often is the classifier correct?

# In[48]:


# use float to perform true division, not integer division
print((TP + TN) / float(TP + TN + FP + FN))


# **Classification Error** : Overall, how often is the classifier incorrect?

# In[49]:


classification_error = (FP + FN) / float(TP + TN + FP + FN)

print(classification_error)


# **Sensitivity** : When the actual value is positive, how often is the prediction correct?

# In[50]:


sensitivity = TP / float(FN + TP)

print(sensitivity)
print(metrics.recall_score(y_test, y_pred_class))


# **Specificity**: When the actual value is negative, how often is the prediction correct?

# In[51]:


specificity = TN / (TN + FP)

print(specificity)


# **False Positive Rate** : When the actual value is negative, how often is the prediction incorrect?

# In[52]:


false_positive_rate = FP / float(TN + FP)

print(false_positive_rate)
#print(1 - specificity)


# **Precision** : When a positive value is predicted, how often is the prediction correct?

# In[53]:


precision = TP / float(TP + FP)

#print(precision)
print(metrics.precision_score(y_test, y_pred_class))


# ### Cross-validation:
# 
# Now instead of a single train/test split, We use K-Fold cross validation to get a better measure of your model's accuracy (K=10).
# 

# In[54]:


from sklearn.model_selection import cross_val_score

dectree = DecisionTreeClassifier(random_state=1)

cv_scores = cross_val_score(dectree, features_final, asd_class, cv=10)

cv_scores.mean()


# ### AUC Score: 
# 
# AUC is the percentage of the ROC plot that is underneath the curve

# In[55]:


# calculate cross-validated AUC
from sklearn.model_selection import cross_val_score
cross_val_score(dectree, features_final, asd_class, cv=10, scoring='roc_auc').mean()


# ### F-beta Score:

# In[56]:


dectree.fit(X_train, y_train)
from sklearn.metrics import fbeta_score
predictions_test = dectree.predict(X_test)
fbeta_score(y_test, predictions_test, average='binary', beta=0.5)


# In[ ]:





# ---
# <a id='2'></a>
# ## (2) Random Forest
# 
# Now I apply a **RandomForestClassifier** instead to see whether it performs better.

# In[57]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
ranfor = RandomForestClassifier(n_estimators=5, random_state=1)
cv_scores = cross_val_score(ranfor, features_final, asd_class, cv=10)
cv_scores.mean()


# In[58]:


ranfor = ranfor.fit(X_train, y_train)


# ## <a id='2'></a>
# ## Predictions and Evaluation of RF Model

# In[59]:


predictions_rf = ranfor.predict(X_test)
acc_rf = accuracy_score(y_true = y_test, y_pred = predictions_rf )
print("Overall accuracy of RFM using test-set is : %f" %(acc_rf*100))


# In[60]:


print(classification_report(y_test, predictions_rf))


# In[61]:


print(confusion_matrix(y_test, predictions_rf))


# AUC Score: AUC is the percentage of the ROC plot that is underneath the curve

# In[62]:


# calculate cross-validated AUC
from sklearn.model_selection import cross_val_score
cross_val_score(ranfor, features_final, asd_class, cv=10, scoring='roc_auc').mean()


# F-beta Score:

# In[63]:


ranfor.fit(X_train, y_train)
from sklearn.metrics import fbeta_score
predictions_test = ranfor.predict(X_test)
fbeta_score(y_test, predictions_test, average='binary', beta=0.5)


# ---
# <a id='3'></a>
# ## (3) SVM
# 
# Next We tried using svm.SVC with a linear kernel and see how well it does in comparison to the decision tree.

# In[64]:


from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

C = 1.0
svc = svm.SVC(kernel='linear', C=C, gamma=2)


# In[65]:


model_svm = SVC()


# In[66]:


model_svm.fit(X_train, y_train)


# <a id='2'></a>
# ## Predictions and Evaluation of SVM Model

# In[67]:


predictions_svm = model_svm.predict(X_test)
acc_svm = accuracy_score(y_true = y_test, y_pred = predictions_svm)
print("Overall accuracy of SVM using test-set is : %f" %(acc_svm*100))


# In[68]:


print(classification_report(y_test, predictions_svm))


# In[69]:


print(confusion_matrix(y_test, predictions_svm))


# In[70]:


cv_scores = cross_val_score(svc, features_final, asd_class, cv=10)

cv_scores.mean()


# AUC Score: AUC is the percentage of the ROC plot that is underneath the curve

# In[71]:


# calculate cross-validated AUC
from sklearn.model_selection import cross_val_score
cross_val_score(svc, features_final, asd_class, cv=10, scoring='roc_auc').mean()


# F-beta Score:

# In[72]:


svc.fit(X_train, y_train)
from sklearn.metrics import fbeta_score
predictions_test = svc.predict(X_test)
fbeta_score(y_test, predictions_test, average='binary', beta=0.5)


# ---
# <a id='4'></a>
# 
# ## (4) K-Nearest-Neighbors (KNN)
# Next, we explore the K-Nearest-Neighbors algorithm with a starting value of K=10. Recall that K is an example of a hyperparameter - a parameter on the model itself which may need to be tuned for best results on your particular data set.

# In[73]:


from sklearn import neighbors
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

knn = neighbors.KNeighborsClassifier(n_neighbors=5)
cv_scores = cross_val_score(knn, features_final, asd_class, cv=10)

cv_scores.mean()


# In[74]:


knn.fit(X_train, y_train)


# <a id='2'></a>
# ## Predictions and Evaluation of kNN Model

# In[75]:


predictions_knn=knn.predict(X_test)
acc_knn = accuracy_score(y_true = y_test, y_pred = predictions_knn)
print("Overall accuracy of kNN using test-set is : %f" %(acc_knn*100))


# In[76]:


print(classification_report(y_test, predictions_knn))


# In[77]:


print(confusion_matrix(y_test, predictions_knn))


# AUC Score: AUC is the percentage of the ROC plot that is underneath the curve

# In[78]:


# calculate cross-validated AUC
from sklearn.model_selection import cross_val_score
cross_val_score(knn, features_final, asd_class, cv=10, scoring='roc_auc').mean()


# F-beta Score:

# In[79]:


knn.fit(X_train, y_train)
from sklearn.metrics import fbeta_score
predictions_test = knn.predict(X_test)
fbeta_score(y_test, predictions_test, average='binary', beta=0.5)


# Choosing K is tricky, so we can't discard KNN until we've tried different values of K. Hence we write a for loop to run KNN with K values ranging from 10 to 50 and see if K makes a substantial difference.
# 

# In[80]:


for n in range(10, 50):
    knn = neighbors.KNeighborsClassifier(n_neighbors=n)
    cv_scores = cross_val_score(knn, features_final, asd_class, cv=10)
    print (n, cv_scores.mean())


# ---
# <a id='5'></a>
# 
# ## (5) Naive Bayes
# 
# Now we tried naive_bayes.MultinomialNB classifier and ask how does its accuracy stack up.

# In[81]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


nb = MultinomialNB()
cv_scores = cross_val_score(nb, features_final, asd_class, cv=10)

cv_scores.mean()


# In[82]:


nb.fit(X_train, y_train)


# <a id='2'></a>
# ## Predictions and Evaluation of NB Model

# In[83]:


predictions_nb=nb.predict(X_test)
acc_nb = accuracy_score(y_true = y_test, y_pred = predictions_nb)
print("Overall accuracy of NB using test-set is : %f" %(acc_nb*100))


# In[84]:


print(classification_report(y_test, predictions_nb))


# In[85]:


print(confusion_matrix(y_test, predictions_nb))


# AUC Score: AUC is the percentage of the ROC plot that is underneath the curve

# In[86]:


# calculate cross-validated AUC
from sklearn.model_selection import cross_val_score
cross_val_score(nb, features_final, asd_class, cv=10, scoring='roc_auc').mean()


# F-beta Score:

# In[87]:


nb.fit(X_train, y_train)
from sklearn.metrics import fbeta_score
predictions_test = nb.predict(X_test)
fbeta_score(y_test, predictions_test, average='binary', beta=0.5)


# ---
# <a id='6'></a>
# 
# ## (6) Logistic Regression
# 
# We've tried all these fancy techniques, but fundamentally this is just a binary classification problem. Try Logisitic Regression, which is a simple way to tackling this sort of thing.

# In[120]:


from sklearn.impute import SimpleImputer


# In[125]:


imputer = SimpleImputer(missing_values= np.nan, strategy='mean')


# In[126]:


imputer.fit(features_final)


# In[131]:


features_final = imputer.transform(features_final)


# In[132]:


features_final_1 = pd.DataFrame(features_final)


# In[135]:


features_final_1.head().isnull().mean()


# In[188]:



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
logreg = LogisticRegression()
cv_scores = cross_val_score(logreg, features_final_1, asd_class, cv=10)
cv_scores.mean()


# In[189]:


logreg.fit(X_train, y_train)


# <a id='2'></a>
# ## Predictions and Evaluation of LR Model

# In[190]:


predictions_logreg = logreg.predict(X_test)
acc_logreg = accuracy_score(y_true = y_test, y_pred = predictions_logreg)
print("Overall accuracy of LR using test-set is : %f" %(acc_logreg*100))


# In[191]:


print(classification_report(y_test, predictions_logreg))


# In[192]:


print(confusion_matrix(y_test, predictions_logreg))


# AUC Score: AUC is the percentage of the ROC plot that is underneath the curve

# In[193]:


# calculate cross-validated AUC
from sklearn.model_selection import cross_val_score
cv_scores_roc = cross_val_score(logreg, features_final_1, asd_class, cv=10, scoring='roc_auc').mean()
cv_scores_roc.mean()


# F-beta Score:

# In[194]:


logreg.fit(X_train, y_train)
from sklearn.metrics import fbeta_score
predictions_test = logreg.predict(X_test)
fbeta_score(y_test, predictions_test, average='binary', beta=0.5)


# ---
# <a id='7'></a>
# 
# ## (7) Linear Discriminant Analysis

# In[95]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

lda = LinearDiscriminantAnalysis()
cv_scores = cross_val_score(lda, features_final, asd_class, cv=10)
cv_scores.mean()


# In[96]:


lda.fit(X_train, y_train)


# <a id='2'></a>
# ## Predictions and Evaluation of LDA Model

# In[97]:


predictions_lda = lda.predict(X_test)
acc_lda = accuracy_score(y_true = y_test, y_pred = predictions_lda)
print("Overall accuracy of LDA using test-set is : %f" %(acc_lda*100))


# In[98]:


print(classification_report(y_test, predictions_lda))


# In[99]:


print(confusion_matrix(y_test, predictions_lda))


# AUC Score: AUC is the percentage of the ROC plot that is underneath the curve

# In[100]:


# calculate cross-validated AUC
from sklearn.model_selection import cross_val_score
cv_scores_roc = cross_val_score(lda, features_final, asd_class, cv=10, scoring='roc_auc').mean()
cv_scores_roc.mean()


# F-beta Score:

# In[101]:


lda.fit(X_train, y_train)
from sklearn.metrics import fbeta_score
predictions_test = lda.predict(X_test)
fbeta_score(y_test, predictions_test, average='binary', beta=0.5)


# ---
# <a id='8'></a>
# ## (8) Quadratic Discriminant Analysis

# In[102]:


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
qda = QuadraticDiscriminantAnalysis()
cv_scores = cross_val_score(qda, features_final, asd_class, cv=10)
cv_scores.mean()


# In[103]:


qda.fit(X_train, y_train)


# <a id='2'></a>
# ## Predictions and Evaluation of QDA Model

# In[104]:


predictions_qda = qda.predict(X_test)
acc_qda = accuracy_score(y_true = y_test, y_pred = predictions_qda)
print("Overall accuracy of QDA using test-set is : %f" %(acc_qda*100))


# In[105]:


print(classification_report(y_test, predictions_qda))


# In[106]:


print(confusion_matrix(y_test, predictions_qda))


# AUC Score: AUC is the percentage of the ROC plot that is underneath the curve

# In[107]:


# calculate cross-validated AUC
from sklearn.model_selection import cross_val_score
cv_scores_roc = cross_val_score(qda, features_final, asd_class, cv=10, scoring='roc_auc').mean()
cv_scores_roc.mean()


# F-beta Score:

# In[108]:


qda.fit(X_train, y_train)
from sklearn.metrics import fbeta_score
predictions_test = qda.predict(X_test)
fbeta_score(y_test, predictions_test, average='binary', beta=0.5)


# 

# ---
# <a id='step8'></a>
# 
# ## Step 8: Conclusion
# 
# 
# After exploring our `ASD` dataset with different kind of learning algorithms, We have arrived into this conclusion that all of our model work extremely well with the data. We have used three different `metric` (such as `accuracy`, `AUC score` and `F-score`) to measure the performance of our models, and it seems like all of the `metric` indicated an almost perfect classification of the ASD cases. 
# I think to build a more accurate model, we need to have access to more larger datasets. Here the number of instances after cleaning the data were not so sufficient enough so that we can claim that this model is optimum. As this dataset is only publicly available from Decemeber 2017, I think not many works that deal with this dataset is available online. In that consideration, our models can serve as benchmark models for any machine learning researcher/practitioner who will be interested to explore this dataset further. With this fact in mind, I think this are very well developed model that can detect ASD in indivisuals with certain given attributes.

# In[109]:


li_x = ['DT','RF','SVM','kNN', 'NB', 'LR', 'LDA', 'QDA']
li_y = [acc_dectree, acc_rf, acc_svm, acc_knn, acc_nb, acc_logreg, acc_lda, acc_qda]


# In[110]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid", color_codes=True)
print(li_y)
sns.barplot(x= li_x, y= li_y)


# #### **Give 18 inputs according to following:** <br>
# 
# A1 -> Does the patient loot at you when you call his/her name? (yes:1, NO:0) <br>
# A2 -> Does the patient have repetitive behaviour? (yes:1, NO:0) <br>
# A3 -> Does the patient point to indicate that he/she wants something? (yes:1, NO:0) <br>
# A4 -> Is the patient facing difficulty realting to people? (yes:1, NO:0) <br>
# A5 -> Does the adult have obsessive interest? (yes:1, NO:0) <br>
# A6 -> Does the patient have social anxiety?  (yes:1, NO:0) <br>
# A7 -> Is there lack of social skills in the patient? (yes:1, NO:0) <br>
# A8 -> Does the patient face difficulty in eye contact? (yes:1, NO:0) <br>
# A9 -> Do the patient have repetitive behaviour? (yes:1, NO:0) <br>
# A10 -> Does the patient have bad executive function? (yes:1, NO:0) <br>
# <br>
# What is the patient relation with who is giving this test? -> Health care professional:1, others:2, parent:3, Relative:4, self:5 <br>
# <br>
# What is the patient's age? -> 18 and more:1, Less than 18:0 <br>
# have you used this app before? -> Yes:1, No:0 <br>
# 
# What is the country of residence of the patient?  -> <br>
# <br>
# United States	0	Sierra Leone	30 <br>
# Brazil	1	Ethiopia	31 <br>
# Spain	2	Viet Nam	32 <br>
# New Zealand	3	Iran	33 <br>
# Bahamas	4	Costa Rica	34 <br>
# Burundi	5	Germany	35 <br>
# Jordan	6	Mexico	36 <br>
# Ireland	7	Armenia	37 <br>
# United Arabs Emirates	8	Iceland	38 <br>
# Afghanistan	9	Nicaragua	39 <br>
# United Kingdom	10	Austria	40 <br>
# South Africa	11	Russia	41 <br>
# Italy	12	AmericanSamoa	42 <br>
# Pakisthan	13	Uruguay	43 <br>
# Egypt	14	Ukraine	44 <br>
# Bangladesh	15	Serbia	45 <br>
# Chile	16	Portugal	46 <br>
# France	17	Malaysia	47 <br>
# China	18	Ecuador	48 <br>
# Australia	19	Niger	49 <br>
# Canada	20	Belhium	50 <br>
# Saudi Arabia	21	Bolivia	51 <br>
# Netherlands	22	Aruba	52 <br>
# Romania	23	Finland	53 <br>
# Sweden	24	Turkey	54 <br>
# Tonga	25	Nepal	55 <br>
# Oman 	26	Indonesia	56 <br>
# India	27	Angola	57 <br>
# Philippines	28	Czech Republic	58 <br>
# Sri Lanka	29	Cyprus	59 <br>
# <br>
# 
# Have u had Autism before?  ->  Yes:1, No:0 <br>
# Is the patient born with jaundice? ->  Yes:1, No:0<br>
# <br>
# What is the ethnicity of the patient? -> <br>
# White-European:0 <br>
# Latino:1 <br>
# Others,others:2 <br>
# Black:3 <br>
# Asian:4 <br>
# Middle Eastern:5 <br>
# Pasifika: 6 <br>
# South Asian:7 <br>
# Hispanic:8 <br>
# Turkish:9 <br>
# 
# What is the gender of patient? -> f:0, m:1

# In[195]:


l = [[1,0,1,1,1,1,0,0,0,1,5,1,0,27,0,0,4,0]]


# In[112]:


predict = dectree.predict(l)


# In[113]:


# print(predict)
if predict==1:
   print("Patient is having ASD.")
else:
    print("Patient is not having ASD.")


# In[196]:


import pickle
pickle.dump(logreg, open('modellr.pkl', 'wb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




