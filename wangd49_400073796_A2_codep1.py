# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 13:14:17 2020

@author: David Wang
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#load dataset

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
#print(cancer.DESCR)
X, t = load_breast_cancer(return_X_y=True)

# split data into trainig and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, t_train, t_test = train_test_split(X, t, test_size = 1/4, random_state = 3796)
# print(X_train.shape) # X_train is 2D, but y_train is 1D
#print(t_train.shape)
M = len(X_test) #number rows in test set
N = len(X_train) #number rows in train set
# print(N, M)

#normalizing the data
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train[:,:] = sc.fit_transform(X_train[:,:])
X_test[:,:] = sc.transform(X_test[:,:])


#class 0  
i0 = np.asarray(np.nonzero(t_train==0)) #indexes where class is 0
#print(i0)
[m,n] = i0.shape
X_train_0 = np.zeros((n,30))
t_train_0 = np.zeros(n)
#print(t_train_0)
for i in range(n):
    X_train_0[i,:] = X_train[i0[0,i],:] 
#print(X_train_0)

#class 1
i1 = np.asarray(np.nonzero(t_train==1)) #indexes where class is 0
#print(i1)
[m,n] = i1.shape
#print(n)
X_train_1 = np.zeros((n,30))
t_train_1 = np.ones(n)
#print(t_train_1)
for i in range(n):
    X_train_1[i,:] = X_train[i1[0,i],:] 
#print(X_train_1)

#plot classes
# plt.scatter(X_train_0[:,0], X_train_0[:,1], color = 'red')
# plt.scatter(X_train_1[:,0], X_train_1[:,1], color = 'blue') #malignant




def gradientdescent():
    new_col=np.ones(N)
    X1_train = np.insert(X_train, 0, new_col, axis=1) # dummy feature was included
    alpha = 1 #learning rate
    #initialize w
    global w
    w = np.array([-10,-10,-10,-10,-10,-10,
              -10,-10,-10,-10,-10,-10,
              -10,-10,-10,-10,-10,-10,
              -10,-10,-10,-10,-10,-10,
              -10,-10,-10,-10,-10,-10,-10]) #initial vetor of weights
    z = np.zeros(N)
    IT = 300 #iterations
    gr_norms = np.zeros(IT) # to store squared norm of gradient at ecah iteration
    cost = np.zeros(IT)  # to store the cost at each iteration
    for n in range(IT):
        z = np.dot(X1_train,w)
        #y = 1/np.logaddexp(0, -z)
        y = 1/(1 + np.exp(-z))
        diff = y-t_train
        gr = np.dot(X1_train.T, np.transpose(diff.T))/N # this is the gradient
        #compute squared norm of the gradient
        gr_norm_sq = np.dot(gr,gr)
        gr_norms[n] = gr_norm_sq
        #update the vector of parameters
        w = w - alpha * gr 
        #compute the cost
        cost[n] = 0
        for i in range(N):
            cost[n] += t_train[i]*np.logaddexp(0, -z[i]) + (1-t_train[i])*np.logaddexp(0,z[i])
        cost[n] = cost[n]/N
    #print(w)
    # print(cost[:5])
    # print(cost[IT-5:IT])
    # print(gr_norms[IT-5:IT])
    
def misclassification(theta):
    #compute test error
    new_col=np.ones(M)
    X1_test = np.insert(X_test, 0, new_col, axis=1) # dummy feature was included
    z = np.dot(X1_test,w) #z=wT*x
    global y
    y = np.zeros(M)
    for i in range(M):
        if(z[i]>=theta):#if getter than theta y is set as positive
            y[i]=1
    u = y - t_test
    #print(u)
    err = np.count_nonzero(u)/M  #mislassification rate
    print("this is misclassfication rate" ,err)
    
def PR():
    TP = 0
    FP = 0
    FN = 0 #setting initial conditions
    for i in range(M): #go through test set to find types of pos
        if (y[i]==1 and t_test[i]==1):
            TP=TP+1
        if (y[i]==1 and t_test[i]==0):
            FP=FP+1
        if (y[i]==0 and t_test[i]==1):
            FN=FN+1
        
    #print(TP,FP,FN)
    P=TP/(TP+FP) # doing P and R calculations
    R=TP/(TP+FN)
    F1=2*(P*R)/(P+R) #doing F1 calculations
    #print(P)
    #print(R)
    print("this is F1 score",F1)
    global arrayP,arrayR
    arrayP=np.append(arrayP,P) #used to plot all points of p, add p to array of p
    arrayR=np.append(arrayR,R) #same for R
    
#function used for manual KNN    
def KNN(k,M,N,trainx,traint,testx,testt): #neighbors,testlength,trainlength, trainx,train t, testx,testt
        global err
        global y
        #initialize arrays, M->test N->train
        dist = np.zeros((M,N)) #2dim array to store distances from test points to trainig points 
        ind = np.zeros((M,N))  #2dim array to store the order after sorting the distances
        u = np.arange(N)       # array of numbers from 0 to N-1
        for j in range(M): #giving each row in ind matrix 0 to N-1
            ind[j,:] = u
        
        #compute distances and sort 
        for j in range(M): #each test point
            for i in range(N): #each training point
                z = trainx[i,:]-testx[j,:] #compare all features of specific test point to training points
                dist[j,i] = np.dot(z,z) #compute a distance value for the specific pair
                #ind[j,:] = np.argsort(dist[j,:])
        #print(dist[:5,:5])
        ind = np.argsort(dist) #make ind matrix sort the distances from smallest to largest and store as index
        #print(ind.shape)
        
        # compute predictions and error with 1NN
        y = np.zeros(M) # initialize array of predictions
        for j in range(M):
            #y[j] = t_train[ind[j,0]]
            accum=0
            for i in range(k):
                accum = accum + traint[ind[j,k-1]] #takes k nearest neighbors
            y[j]=accum/k#average values
        #print(y)
        #print(t_test)
        z = y - testt
        #print(z)
        err = np.count_nonzero(z)/M  #mislassification rate
        print(err)
        
def KfoldKNN(neighbors,splits):#function used to perform Kfold KNN manually using KNN function
    print("for",neighbors,"nearest neighbors")
    from sklearn.model_selection import KFold
    kf= KFold(n_splits=splits, random_state=3796, shuffle=True)
    #print(kf)
    total = 0 #initial value for total err
    i=1#keeping track of folds
    global average
    global averageerror
    for train_index, test_index in kf.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        KX_train, KX_test = X[train_index], X[test_index]
        Kt_train, Kt_test = t[train_index], t[test_index]
        lentrain =len(KX_train)
        lentest = len(KX_test)
        KNN(neighbors,lentest,lentrain,KX_train,Kt_train,KX_test,Kt_test)
        total=total+err
    averageerror=total/splits
    print("average error:",averageerror)

def SciKNN(folds, N): #function used to perform Kfold KNN with scikit
    print("for",N,"nearest neighbors")
    from sklearn.model_selection import KFold
    kf= KFold(n_splits=folds, random_state=3796, shuffle=True) #randomly shuffling X and splitting to 5 sets
    #print(kf)
    total = 0 #initial value for total err
    i=1#keeping track of folds
    global average
    for train_index, test_index in kf.split(X): #taking index's of X and assigning values
        #print("TRAIN:", train_index, "TEST:", test_index) 
        KX_train, KX_test = X[train_index], X[test_index]
        Kt_train, Kt_test = t[train_index], t[test_index]
        from sklearn.neighbors import KNeighborsClassifier
        neigh = KNeighborsClassifier(n_neighbors=N)
        neigh.fit(KX_train,Kt_train)
        predictions=neigh.predict(KX_test)
        score=neigh.score(KX_test,Kt_test)
        err=1-score
        total=total+err
        print("scikit misclassification rate for k=",i,":",err)
        i=i+1
    average=total/5
    print("average misclassification rate:",average)
        
    
arrayP=[] #making empty arrays to store data points for thetas
arrayR=[]    
#------------------------------------------------------------------------------------------------------    
# #MANUAL LOGISTIC REGRESIION

# gradientdescent()
# #just the 0 theta
# misclassification(0)
# PR()

# #graph stuff

# for i in range(-20,100,10):
#     misclassification(i)
#     PR()
# plt.plot(arrayR,arrayP) #print all data pairs of PvsR

#------------------------------------------------------------------------------------------------------
#Scikit learn logstic regression
# from sklearn.linear_model import LogisticRegression
# logisticreg=LogisticRegression()
# logisticreg.fit(X_train,t_train) #training model
# predictions=logisticreg.predict(X_test)#testing model
# score = logisticreg.score(X_test,t_test)
# print("scikit misclassification rate",1-score)

# #PR curve
# from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import plot_precision_recall_curve
# disp = plot_precision_recall_curve(logisticreg.fit(X_train,t_train),X_test,t_test) 
# #logisticreg.fit is the binary classifer

# from sklearn.metrics import f1_score
# F1score=f1_score(t_test, predictions, average='binary')#calculate F1 score
# print("scikit F1 score", F1score)

# #------------------------------------------------------------------------------------------------------
# #MANUAL K NN 
# smallesterror=1 #set initial error to 1

# for i in range(1,6): #run through 5 neighbors and update smalest error
#     KfoldKNN(i,5)
#     if(smallesterror>averageerror):
#         smallesterror=averageerror
#         iteration=i
# print("iteration",iteration, "has the smallest error:", smallesterror)
# print("Best classifier test error:")
# KNN(iteration, M,N,X_train,t_train,X_test,t_test)
# PR()#to find F1 score


#------------------------------------------------------------------------------------------------------

# ##Scikit KNN
smallesterr=1#set initial error to 1

    
for p in range(1,6):    
    SciKNN(5,p)
    if (smallesterr>average):#update the smallest error
        smallesterr=average
        iteration=p
print ("iteration",iteration,"has the smallest error:",smallesterr)   

from sklearn.neighbors import KNeighborsClassifier ##using best classifier on training set and test set
bestfit=KNeighborsClassifier(n_neighbors=iteration)
bestfit.fit(X_train,t_train)
bestpredictions = bestfit.predict(X_test)
bestscore=bestfit.score(X_test,t_test)
besterr=1-bestscore
print("scikit best classifier test error:",besterr)

from sklearn.metrics import f1_score
F1score=f1_score(t_test, bestpredictions, average='binary')#calculate F1 score
print("scikit F1 score", F1score)