#!/usr/bin/env python
# coding: utf-8
#CSE 572 Assignmen 3
#Ziming Dong

import pickle
from numpy import *
import operator
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
from csv import writer
from csv import reader
from itertools import zip_longest
from sklearn.neighbors import KNeighborsClassifier

def CSVlist(filename):
    File=[]
    with open(filename,newline='',encoding='utf-8-sig')as file:
        data = csv.reader(file)
        for row in data:
            File.append(row)
    return File

# Please change the test file name here!!!!!!!
# Replace the name mealData2.csv by the test file name and path
# Then Run the code, a csv file should be generated automatically.
Data=CSVlist('mealData1.csv')
def clean(data):
    x = []
    for i in range (len(data)):
        data[i] = data[i][::-1]
        data[i] = data[i][:30]
        if (len(data[i])!= 30):
            x.append(i)
        elif 'NaN' in data[i]:
            x.append(i)      
    for j in range (len(x),0,-1):
        del data[x[j-1]]
    return data
Data=clean(Data)
dfmeal = pd.DataFrame(Data)

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

#feature 1 cgmmax-cgm0
feature1=[]
feature1_norm=[]
f1_result=0
cgmlist=[]
for i in range(len(Data)):
    f1=dfmeal.iloc[i]
    for i in f1:
        cgmlist.append(i)      
    m=max(cgmlist)
    began=f1[5]
    f1_result=np.array((int(m)-int(began))/int(began))
    feature1.append(f1_result)  
feature1_norm=np.array(normalize(feature1))
feature1=np.array(feature1)

#feature 3 interquartile range
feature3=[]
feature3_norm=[]
f3_result=0
for i in range(0,len(Data)):
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
for i in range(len(Data)):
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

matrix12=np.vstack((feature1_norm,feature3_norm))
matrix12=matrix12.T
# matrix1=matrix1.T
# matrix1.shape
# feature_matrix1=np.vstack((matrix1,feature4_norm))
# feature_matrix=feature_matrix1.T
# feature_matrix=np.concatenate((feature_matrix1,feature4_norm),axis=1)
# feature_matrix.shape
feature_matrix=np.column_stack([matrix12,feature4_norm])
# feature_matrix=np.column_stack([feature_matrix1,feature5])


km_final=[]
feature = pickle.load(open('feature.pickle', 'rb'))
km_label = pickle.load(open('km_label.pickle', 'rb'))
for i in range(len(Data)):
    f1=feature_matrix[i]
    model = KNeighborsClassifier(n_neighbors=7)
    model.fit(feature,km_label)
    result=model.predict([f1])
    km_final.append(result)
# km_final

db_final=[]
db_label = pickle.load(open('db_label.pickle', 'rb'))
for i in range(len(Data)):
    f1=feature_matrix[i]
    model = KNeighborsClassifier(n_neighbors=7)
    model.fit(feature,db_label)
    result=model.predict([f1])
    db_final.append(result)



df = pd.DataFrame(list())
df.to_csv('labels.csv')

export_data = zip(km_final,db_final)
with open('labels.csv', 'w', encoding='utf-8-sig', newline='') as myfile:
      wr = csv.writer(myfile)
      wr.writerows(export_data)
    

df = pd.read_csv('labels.csv', header=None)
df.rename(columns={0: 'K-Means',1:'DB-Scan'}, inplace=True)
df.to_csv('labels.csv', index=False)

