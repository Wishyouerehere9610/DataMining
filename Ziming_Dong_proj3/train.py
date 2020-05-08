#!/usr/bin/env python
# coding: utf-8

#CSE 572 Assignmen 3
#Ziming Dong
from __future__ import division
import matplotlib.pyplot as plt
import pandas as pd
import os,glob
import warnings
import numpy as np
import seaborn
from scipy.fftpack import fft,ifft
from scipy.stats import kurtosis,skew
from sklearn.decomposition import PCA
from numpy import linalg as LA
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
import pickle
import csv
import itertools
import math
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import collections, numpy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

#Save the csv files data into list for meal and nomeal data.
meal=[]
for i in range (5):
    with open(r'mealData'+str(i+1)+'.csv','rt')as mealfile:
        data = csv.reader(mealfile)
        for row in itertools.islice(data,50):
            meal.append(row)
            

for i in range(len(meal)):
    meal[i] = meal[i][::-1]
    meal[i] = meal[i][:30]

nan=[]
for i in range(len(meal)):
    if 'NaN' in meal[i]:
        nan.append(i)

l=[]
for i in range(len(meal)):
    if len(meal[i])!=30:
        l.append(i)


noise_index=list(set(nan) | set(l))


def clean(data):
    x = []
    for i in range (len(data)):

        data[i] = data[i][:30]
        if (len(data[i])!= 30):
            x.append(i)
        elif 'NaN' in data[i]:
            x.append(i)      
    for j in range (len(x),0,-1):
        del data[x[j-1]]
    return data
Meal=clean(meal)

amount=[]
for i in range (5):
    with open(r'mealAmountData'+str(i+1)+'.csv','rt')as mealfile:
        data = csv.reader(mealfile)
        for row in itertools.islice(data,50):
            amount.append(row)

Amount = np.array(amount) 
# Noise = np.array(noise_index)

new_amount=np.delete(Amount,noise_index)

#Define a function to normalization value to put into PCA.
def normalize( lst ):
    newlst = [] # the variable to store normalized values
    minx = lst[0] # variable to store minx
    maxx = lst[0] # variable for maxx
    for k in lst: # loop to find min and max in input list
        if k > maxx:
            maxx = k # finding the maxx
        if k < minx:
            minx = k # finding the minx
    for k in lst:
        newlst = newlst + [float((k-minx))/(maxx-minx)] # computing the normalized value
    return newlst

dfmeal = pd.DataFrame(Meal)

#feature 1 cgmmax-cgm0
feature1=[]
feature1_norm=[]
f1_result=0
cgmlist=[]
for i in range(211):
    f1=dfmeal.iloc[i]
    for i in f1:
        cgmlist.append(int(i))      
    m=max(cgmlist)
    began=f1[5]
    f1_result=np.array((int(m)-int(began))/int(began))
    feature1.append(f1_result)  
feature1_norm=np.array(normalize(feature1))
feature1=np.array(feature1)


# feature 2 (ct=0-ct=w)/w
feature2=[]
feature2_norm=[]
f2_result=0
for i in range(211):
    f1=dfmeal.iloc[i]
    w=f1[24]
    began=f1[5]
    f2_result=abs((int(began)-int(w))/19)
    feature2.append(f2_result)
feature2_norm=np.array(normalize(feature2))
feature2=np.array(feature2)

#feature 3 interquartile range
feature3=[]
feature3_norm=[]
f3_result=0
for i in range(0,211):
    q=np.asarray(dfmeal.iloc[i],dtype=np.float32)
    q1_x=np.quantile(q,0.25,interpolation='midpoint')
    q3_x=np.quantile(q,0.75,interpolation='midpoint')
    f3_result=np.array(q3_x-q1_x)
    feature3.append(f3_result)
feature3_norm=np.array(normalize(feature3))
feature3=np.array(feature3)

# feature 4 (ct=0-ct=w)/w windows
feature4=[]
norm=[]
result=[]
f4_result=0
for i in range(211):
    f4=dfmeal.iloc[i]
    for i in range(5,24,5):
        began=f4[i]
        w=f4[i+5]
        f4_result=float(abs((int(began)-int(w))/5))
        result.append(f4_result)
    feature4.append(result)
    result=[]
feature4=np.array(feature4)
feature4_norm=[[(x-min(l))/(max(l)-min(l)) for x in l] for l in feature4]
feature4_norm=np.array(feature4_norm)

#feature5 fft
feature5=[]
feature5_norm=[]
for i in range(0,211):
    x=np.linspace(0,29,30)
    y=dfmeal.iloc[i]
    yy=abs(fft(y))              
    yy1=np.delete(yy,0)
    yy2=np.unique(yy1)
    Max5=np.partition(yy2,-5)[-5:]
    normMax5=Max5/Max5.max(axis=0)
    feature5.append(Max5)
    feature5_norm.append(normMax5)
feature5=np.array(feature5)

matrix12=np.vstack((feature1_norm,feature3_norm))
matrix12=matrix12.T
# matrix1=matrix1.T
# matrix1.shape
# feature_matrix1=np.vstack((matrix1,feature4_norm))
# feature_matrix=feature_matrix1.T
# feature_matrix=np.concatenate((feature_matrix1,feature4_norm),axis=1)
# feature_matrix.shape
new_feature_matrix=np.column_stack([matrix12,feature4_norm])
# feature_matrix=np.column_stack([feature_matrix1,feature5])

# pca=PCA(n_components=6)

# principalComponents=pca.fit_transform(feature_matrix)

# # principalDf = pd.DataFrame(data = principalComponents
# #              , columns = ['principal component 1', 'principal component 2','principal component 3',
# #                            'principal component 4','principal component 5','principal component 6','principal component 7',
# #                           'principal component 8','principal component 9','principal component 10','principal component 11'
# #                          ,'principal component 12','principal component 13'])
# print(pca.explained_variance_ratio_)
# values=abs(pca.components_)
# print(   )
# print(values[0])

# new_feature_matrix=np.delete(feature_matrix,[0,5],axis=1)
# new_feature_matrix.shape

label=[]
for i in range(211):
    b=new_amount[i]
    if int(b)==0:
        label.append(1)
    if 0<int(b)<=20:
        label.append(2)
    if 21<=int(b)<=40:
        label.append(3)
    if 41<=int(b)<=60:
        label.append(4)
    if 61<=int(b)<=80:
        label.append(5)
    if 81<=int(b)<=100:
        label.append(6)

bin_label=np.array(label)

# print(bin_label)
# print(      )
# print(collections.Counter(bin_label))


#-----------------------------------------------K-means Began--------------------------------------------------------
def mf(List): 
    return max(set(List), key = List.count)

def validation(list_1,list_2):
    return sorted(list_1) == sorted(list_2)

def newfeature(indx_list,new_feature_matrix):
    new_feature=[new_feature_matrix[i] for i in indx_list]
#     for i in indx_list:
#         f4=new_feature_matrix[i]
#     new_feature.append(f4)
    new_feature=np.array(new_feature)
#     print(new_feature.shape)
    return new_feature

def kmsecond(combine_feature,bin_label,n):
    kmean=KMeans(n_clusters=n,random_state=0)
    kmean.fit(combine_feature[:,1:8])
    kmlabel=kmean.labels_
    cluster_n=[]
    cluster=[]
    bin_1=[] 
    bin_2=[] 
    bin_3=[] 
    bin_4=[] 
    bin_5=[] 
    bin_6=[]
    or_index=[]
    result=[]
    for j in range(n):
        cluster_n=[i for i in range(len(kmlabel)) if kmlabel[i]==j]
        cluster.append(cluster_n)
        result_label=[bin_label[i] for i in cluster[j]]
        result_n=(max(set(result_label),key=result_label.count)) 
        result.append(result_n)
        if result_n==1:
            for i in cluster_n:
                or_index.append(int(combine_feature[i][0])) 
            bin_1=bin_1+[i for i in or_index]
            or_index=[]
        if result_n==2:
            for i in cluster_n:
                or_index.append(int(combine_feature[i][0])) 
            bin_2=bin_2+[i for i in or_index]
            or_index=[]
        if result_n==3:
            for i in cluster_n:
                or_index.append(int(combine_feature[i][0])) 
            bin_3=bin_3+[i for i in or_index]
            or_index=[]
        if result_n==4:
            for i in cluster_n:
                or_index.append(int(combine_feature[i][0])) 
            bin_4=bin_4+[i for i in or_index]
            or_index=[]
        if result_n==5:
            for i in cluster_n:
                or_index.append(int(combine_feature[i][0])) 
            bin_5=bin_5+[i for i in or_index]
            or_index=[]
        if result_n==6:
            for i in cluster_n:
                or_index.append(int(combine_feature[i][0])) 
            bin_6=bin_6+[i for i in or_index]
            or_index=[]
    return result,bin_1,bin_2,bin_3,bin_4,bin_5,bin_6

def km(new_feature,bin_label,n):
    kmean=KMeans(n_clusters=n,random_state=0)
    kmean.fit(new_feature)
    kmlabel=kmean.labels_
    cluster_n=[]
    cluster=[]
    result_label=[]
    bin_1=[]
    bin_2=[]
    bin_3=[]
    bin_4=[]
    bin_5=[]
    bin_6=[]
    result=[]
    for j in range(n):
        cluster_n=[i for i in range(len(kmlabel)) if kmlabel[i]==j]
        cluster.append(cluster_n)
        result_label=[bin_label[i] for i in cluster[j]]
        result_n=(max(set(result_label),key=result_label.count))
        result.append(result_n)
        if result_n==1: 
            bin_1=bin_1+[i for i in cluster_n]
        if result_n==2: 
            bin_2=bin_2+[i for i in cluster_n]
        if result_n==3: 
            bin_3=bin_3+[i for i in cluster_n]
        if result_n==4: 
            bin_4=bin_4+[i for i in cluster_n]
        if result_n==5: 
            bin_5=bin_5+[i for i in cluster_n]
        if result_n==6: 
            bin_6=bin_6+[i for i in cluster_n]
    return result,bin_1,bin_2,bin_3,bin_4,bin_5,bin_6

f_n=105
first_result=km(new_feature_matrix,bin_label,f_n)
# for i in range(1,7):
#     print(i," ",len(first_result[i]))

bin1_index=first_result[1]
bin2_index=first_result[2]
bin3_index=first_result[3]
bin4_index=first_result[4]
bin1_feature=newfeature(bin1_index,new_feature_matrix)
bin2_feature=newfeature(bin2_index,new_feature_matrix)
bin3_feature=newfeature(bin3_index,new_feature_matrix)
bin4_feature=newfeature(bin4_index,new_feature_matrix)
bin1_IndexArray=np.array(bin1_index)
bin2_IndexArray=np.array(bin2_index)
bin3_IndexArray=np.array(bin3_index)
bin4_IndexArray=np.array(bin4_index)
combine_bin1=np.column_stack([bin1_IndexArray,bin1_feature])
combine_bin2=np.column_stack([bin2_IndexArray,bin2_feature])
combine_bin3=np.column_stack([bin3_IndexArray,bin3_feature])
combine_bin=np.column_stack([bin4_IndexArray,bin4_feature])

# print()
# s_n=15
# split_bin1=kmsecond(combine_bin1,bin_label,s_n)
# for i in range(1,7):
#     print(i," ",len(split_bin1[i]))

# s_n=20
# split_bin1=kmsecond(combine_bin1,bin_label,s_n)
# for i in range(1,7):
#     print(i," ",len(split_bin1[i]))

# bin1_total=split_bin1[1]
# bin2_total=first_result[2]+split_bin1[2]
# bin3_total=first_result[3]+split_bin1[3]
# bin4_total=first_result[4]+split_bin1[4]
# bin5_total=first_result[5]+split_bin1[5]
# bin6_total=first_result[6]+split_bin1[6]

# d_n=20
# split_bin2=kmsecond(combine_bin2,bin_label,d_n)
# for i in range(1,7):
#     print(i," ",len(split_bin2[i]))
# print(split_bin2[5])

# f_n=23
# split_bin4=kmsecond(combine_bin4,bin_label,f_n)
# for i in range(1,7):
#     print(i," ",len(split_bin4[i]))
# print(split_bin4[5])

bin1_total=first_result[1]

bin2_total=first_result[2]

bin3_total=first_result[3]

bin4_total=first_result[4]

bin5_total=first_result[5]

bin6_total=first_result[6]

km_result=[None] * 211
for i in bin1_total:
    km_result[i]=1
for i in bin2_total:
    km_result[i]=2
for i in bin3_total:
    km_result[i]=3
for i in bin4_total:
    km_result[i]=4
for i in bin5_total:
    km_result[i]=5
for i in bin6_total:
    km_result[i]=6

bin_label=list(bin_label)

# len(km_result)
km_acc=sum(1 for x,y in zip(km_result,bin_label) if x == y) / len(bin_label)
# km_ac

KFsplit=KFold(n_splits=5)
KFsplit.get_n_splits(new_feature_matrix)
km_test=[]
km_kf_acc=[]
print()
print("--------------------------------K-means Began----------------------------")
print("The K-means clustering result accurcay compare to ground_truth is: ",km_acc)
print()
for train_index,test_index in KFsplit.split(new_feature_matrix):
    X_train = new_feature_matrix[train_index]
    km_label=[]
    bin_truth=[]
    for i in train_index:
        km_label.append(km_result[i])
    for j in test_index:
        bin_truth.append(bin_label[j])
    X_test = new_feature_matrix[test_index]
    
    km_test=[]
    for k in range(len(X_test)):
        f1=X_test[k]
        model = KNeighborsClassifier(n_neighbors=15)
        model.fit(X_train,km_label)
        result=model.predict([f1])
        km_test.append(result)
#     print(len(km_test))
#     print(len(bin_truth))
#     print(bin_truth)
    acc_km=sum(i == j for i, j in zip(km_test, bin_truth)) / len(bin_truth)
    km_kf_acc.append(acc_km)
    kmeans = KMeans(n_clusters=105, max_iter=1000,random_state=0).fit(X_train)
    sse= kmeans.inertia_
    print("SSE for each iteration:",sse)
    print("The Accuray for k-means from this k-fold validation is: ",acc_km)
    print()
    km_test=[]
# print("avg ",sum(km_kf_acc)/len(km_kf_acc))



#-----------------------------------------------DBSCAN Began--------------------------------------------------------
clustering=DBSCAN(eps=0.373,min_samples=5).fit(new_feature_matrix)
dblabel=clustering.labels_
# print(dblabel.count[1])
# print(dblabel)
# print(      )
# print(collections.Counter(dblabel))

dbindex_mone=[]
dbindex_zero=[]
dbindex_one=[]
dbindex_two=[]
# dbindex_three=[]
# dbindex_four=[]
for i in range(len(dblabel)):
    if dblabel[i]==1:
        dbindex_mone.append(i)
    if dblabel[i]==0:
        dbindex_zero.append(i)
    if dblabel[i]==-1:
        dbindex_one.append(i)
    if dblabel[i]==2:
        dbindex_two.append(i)
#     if dblabel[i]==3:
#         dbindex_three.append(i)
#     if dblabel[i]==4:
#         dbindex_four.append(i)

def transfer(dbindex,bin_label):
    db_bin1=[]
    db_bin2=[]
    db_bin3=[]
    db_bin4=[]
    db_bin5=[]
    db_bin6=[]
    for i in dbindex:
        if bin_label[i]==1:
            db_bin1.append(i)
        if bin_label[i]==2:
            db_bin2.append(i)
        if bin_label[i]==3:
            db_bin3.append(i)
        if bin_label[i]==4:
            db_bin4.append(i)
        if bin_label[i]==5:
            db_bin5.append(i)
        if bin_label[i]==6:
            db_bin6.append(i)
    return db_bin1,db_bin2,db_bin3,db_bin4,db_bin5,db_bin6

db_zero_result=transfer(dbindex_zero,bin_label)
db_one_result=transfer(dbindex_one,bin_label)
db_two_result=transfer(dbindex_two,bin_label)
# db_three_result=transfer(dbindex_three,bin_label)
# db_four_result=transfer(dbindex_four,bin_label)
# for i in range(6):
#     print(i+1," ",len(db_zero_result[i]))
# print()
# for i in range(6):
#     print(i+1," ",len(db_one_result[i]))

db_split_feature=newfeature(dbindex_mone,new_feature_matrix)

combine_mone=np.column_stack([dbindex_mone,db_split_feature])
db_n=60
db_split_mone=kmsecond(combine_mone,bin_label,db_n)
# for i in range(1,7):
#     print(i," ",len(db_split_mone[i]))

zero_one_bin1=db_zero_result[0]+db_one_result[0]+db_two_result[0]
zero_one_bin2=db_zero_result[1]+db_one_result[1]+db_two_result[1]
zero_one_bin3=db_zero_result[2]+db_one_result[2]+db_two_result[2]
zero_one_bin4=db_zero_result[3]+db_one_result[3]+db_two_result[3]
zero_one_bin5=db_zero_result[4]+db_one_result[4]+db_two_result[4]
zero_one_bin6=db_zero_result[5]+db_one_result[5]+db_two_result[5]

db_total_bin1=db_split_mone[1]+zero_one_bin1
db_total_bin2=db_split_mone[2]+zero_one_bin2
db_total_bin3=db_split_mone[3]+zero_one_bin3
db_total_bin4=db_split_mone[4]+zero_one_bin4
db_total_bin5=db_split_mone[5]+zero_one_bin5
db_total_bin6=db_split_mone[6]+zero_one_bin6


db_result=[None] * 211
for i in db_total_bin1:
    db_result[i]=1
for i in db_total_bin2:
    db_result[i]=2
for i in db_total_bin3:
    db_result[i]=3
for i in db_total_bin4:
    db_result[i]=4
for i in db_total_bin5:
    db_result[i]=5
for i in db_total_bin6:
    db_result[i]=6
db_acc=sum(1 for x,y in zip(db_result,bin_label) if x == y) / len(bin_label)
db_acc


KFsplit=KFold(n_splits=5)
KFsplit.get_n_splits(new_feature_matrix)
db_test=[]
db_kf_acc=[]

print()
print("--------------------------------DBSCAN Began----------------------------")
print("The DBSCAN clustering result accurcay compare to ground_truth is: ",db_acc)
print()
for train_index,test_index in KFsplit.split(new_feature_matrix):
    X_train = new_feature_matrix[train_index]
    db_label=[]
    bin_truth=[]
    for i in train_index:
        db_label.append(db_result[i])
    for j in test_index:
        bin_truth.append(bin_label[j])
    X_test = new_feature_matrix[test_index]
    
    db_test=[]
    for k in range(len(X_test)):
        f1=X_test[k]
        model = KNeighborsClassifier(n_neighbors=12)
        model.fit(X_train,db_label)
        result=model.predict([f1])
        db_test.append(result)
#     print(len(db_test))
#     print(len(bin_truth))
#     print(bin_truth)
    acc_db=sum(i == j for i, j in zip(db_test, bin_truth)) / len(bin_truth)
#     kmeans = KMeans(n_clusters=27, max_iter=1000,random_state=0).fit(X_train)
#     sse= kmeans.inertia_
#     print("acc:",acc_db)
    db_kf_acc.append(acc_db)
    print("The Accuray for db_scan from this k-fold validation is: ",acc_db)
    print()
    db_test=[]
# print("avg ",sum(db_kf_acc)/len(db_kf_acc))


#pickle file
pickle.dump(km_result, open('km_label.pickle', 'wb'))
pickle.dump(db_result, open('db_label.pickle', 'wb'))
pickle.dump(new_feature_matrix, open('feature.pickle', 'wb'))


