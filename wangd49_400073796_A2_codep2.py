# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 12:30:34 2020

@author: David Wang
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('data/spambase.data')
X = dataset.iloc[:, :-1].values
t = dataset.iloc[:, -1].values

#splitting test and training data
from sklearn.model_selection import train_test_split
X_train, X_test, t_train, t_test = train_test_split(X, t, test_size = 1/3, random_state = 3796)
# print(X_train.shape) # X_train is 2D, but y_train is 1D
#print(t_train.shape)
M = len(X_test) #number rows in test set
N = len(X_train) #number rows in train set
#print(N, M)
smallesterr=1 # keep track of error for DEcision trees
cross_error_array=[]#keep track of cross error for max leaves
j_array=[] #keep track leaves

def SciDecisionTree(folds, N): #function used to perform Kfold KNN with scikit
    #print("for",N,"leaves")
    from sklearn.model_selection import KFold
    kf= KFold(n_splits=folds, random_state=3796, shuffle=True) #randomly shuffling X and splitting to 5 sets
    #print(kf)
    total = 0 #initial value for total err
    i=1#keeping track of folds
    global average
    global cross_error_array
    for train_index, test_index in kf.split(X): #taking index's of X and assigning values
        #print("TRAIN:", train_index, "TEST:", test_index) 
        KX_train, KX_test = X[train_index], X[test_index]
        Kt_train, Kt_test = t[train_index], t[test_index]
        from sklearn.tree import DecisionTreeClassifier
        from sklearn import tree
        clf=tree.DecisionTreeClassifier(max_leaf_nodes=(N),random_state=(3796))
        clf=clf.fit(X_train, t_train) #train model
        #tree.plot_tree(clf) #plot the tree
        clf.predict(X_test) # predict on test set
        score=clf.score(X_test,t_test) 
        error = 1-score
        total=total+error
        #print("scikit misclassification rate for k=",i,":",error)
        i=i+1
    average=total/5
    cross_error_array=np.append(cross_error_array, average)#putting all cross fold error in array
    #print("average misclassification rate:",average)

#----------------------------------------------------------------
#1
for j in range(2,400):    
    SciDecisionTree(5,j)
    if (smallesterr>average):#update the smallest error
        smallesterr=average
        iteration=j
    j_array=np.append(j_array,j)
print ("this many leaves:",iteration,"has the smallest error:",smallesterr)
#----plot---
plt.figure(1)
plt.plot(j_array,cross_error_array)
plt.xlabel('number of leaves')
plt.ylabel('error')
#-------
smallest_err_array=[]
p_array=[]
for p in range(100,1001,100):
    smallest_err_array=np.append(smallest_err_array,smallesterr)
    p_array=np.append(p_array,p)
    
plt.figure(2)    
plt.plot(p_array,smallest_err_array, label='decision tree flat error')

#------------------------------------------------------------------
#bagging
bagging_error_array=[]
bagging_predictor_array=[]
from sklearn.ensemble import BaggingClassifier

for p in range(100,1001,100):
    clfbagging = BaggingClassifier(n_estimators=p,random_state=(3796)).fit(X_train, t_train)
    scorebagging=clfbagging.score(X_test,t_test)
    errorbagging=1-scorebagging
    bagging_error_array=np.append(bagging_error_array,errorbagging)
    bagging_predictor_array=np.append(bagging_predictor_array,p)
    
plt.plot(bagging_predictor_array,bagging_error_array, label='bagging classifier')

# #------------------------------------------------------------------------
#random forest
random_error_array=[]
random_predictor_array=[]
from sklearn.ensemble import RandomForestClassifier

for p in range(100,1001,100):
    clfrandom=RandomForestClassifier(n_estimators=p,random_state=(3796)).fit(X_train,t_train)
    scorerandom=clfrandom.score(X_test,t_test)
    errorrandom=1-scorerandom
    random_error_array=np.append(random_error_array, errorrandom)
    random_predictor_array=np.append(random_predictor_array, p)

plt.plot(random_predictor_array, random_error_array, label = 'random forest classifier')

#------------------------------------------------------------------------
#stump case max_depth =1
adaboost_error_array=[]
adaboost_predictor_array=[]
from sklearn.ensemble import AdaBoostClassifier
for p in range(100,1001,100):
    clfadaboost=AdaBoostClassifier(n_estimators=p,random_state=(3796)).fit(X_train, t_train)
    scoreadaboost=clfadaboost.score(X_test,t_test)
    erroradaboost=1-scoreadaboost
    adaboost_error_array=np.append(adaboost_error_array, erroradaboost)
    adaboost_predictor_array=np.append(adaboost_predictor_array, p)

plt.plot(adaboost_predictor_array, adaboost_error_array,label='adaboost with stump decision')

#------------------------------------------------------------------------
# max leafnode =10 case

adaboost_error_array1=[]
adaboost_predictor_array1=[]
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
for p in range(100,1001,100):
    clfadaboost1=AdaBoostClassifier(n_estimators=p,random_state=(3796),base_estimator=(DecisionTreeClassifier(max_leaf_nodes=10))).fit(X_train, t_train)
    scoreadaboost1=clfadaboost1.score(X_test,t_test)
    erroradaboost1=1-scoreadaboost1
    adaboost_error_array1=np.append(adaboost_error_array1, erroradaboost1)
    adaboost_predictor_array1=np.append(adaboost_predictor_array1, p)

plt.plot(adaboost_predictor_array1, adaboost_error_array1, label= 'adaboost with 10 max leaf nodes')

#----------------------------------------------------
#no restriction case

adaboost_error_array2=[]
adaboost_predictor_array2=[]
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
for p in range(100,1001,100):
    clfadaboost2=AdaBoostClassifier(n_estimators=p,random_state=(3796),base_estimator=(DecisionTreeClassifier())).fit(X_train, t_train)
    scoreadaboost2=clfadaboost2.score(X_test,t_test)
    erroradaboost2=1-scoreadaboost2
    adaboost_error_array2=np.append(adaboost_error_array2, erroradaboost2)
    adaboost_predictor_array2=np.append(adaboost_predictor_array2, p)

adaboost2=plt.plot(adaboost_predictor_array2, adaboost_error_array2,label='adaboost no restrictions')

#-----------------------------------------------------------------------
plt.legend()
plt.xlabel('number of estimators')
plt.ylabel('error')



                                      
