
#Ziming Dong
#CSE 572 Assignment 1
#02/10/2020

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

# plot the first cell array 
a = pd.read_csv("CGMDatenumLunchPat1.csv")
b = pd.read_csv("CGMSeriesLunchPat1.csv")
plt.figure()
plt.plot(a.iloc[0],b.iloc[0])
plt.show()


# Combine the five csv files and clean the data
path= "/Users/dongziming/Desktop/data mining/assignment 1/DataFolder"

all_files = glob.glob(os.path.join(path,"CGMSeriesLunch*.csv"))
all_df = []
for f in all_files:
    df = pd.read_csv(f, sep=',')
    df['file'] = f.split('/')[-1]
    all_df.append(df)    
merged_df = pd.concat(all_df, ignore_index=True, sort=True)

#Drop the extra time series for every subject.
total=merged_df.drop(['cgmSeries_42','cgmSeries_41','cgmSeries_40','cgmSeries_39','cgmSeries_38','cgmSeries_37','cgmSeries_36','cgmSeries_35','cgmSeries_34','cgmSeries_33','cgmSeries_32','cgmSeries_31'], axis=1)

#Drop the missing value array.
df1=total.dropna()

final=df1.iloc[:,::-1]
nofrows= final.shape[0]
print (nofrows)

final=final.drop(columns="file")

#Feature 1 Ploynomial curve fitting
z_total=[]
z_norm=[]
for i in range(0,186):
    yaxis=np.asarray(final.iloc[i])
    x=np.arange(30)
    z1=np.polyfit(x,yaxis,5)
    z2=z1/z1.max(axis=0)
    z_total.append(z1)
    z_norm.append(z2)
#print value of ploynomial curve fitting
z_total

#Feature 2 FFT
f_total=[]
f_norm=[]
for i in range(0,186):
    x=np.linspace(0,29,30)
    y=final.iloc[i]
    yy=abs(fft(y))              
    yy1=np.delete(yy,0)
    yy2=np.unique(yy1)
    Max5=np.partition(yy2,-5)[-5:]
    normMax5=Max5/Max5.max(axis=0)
    f_total.append(Max5)
    f_norm.append(normMax5)
#print value of FFT    
f_total

#feature 3 skewness 
s_total=[]
for i in range(0,186):
    s1=np.array(skew(final.iloc[i]))
    s_total.append(s1)
#Print value of skewness
s_total

s_avg=sum(s_total)/len(s_total)

#feature 4 interquartile range
q_total=[]
for i in range(0,186):
    q=final.iloc[i]
    q1_x=np.quantile(q,0.25,interpolation='midpoint')
    q3_x=np.quantile(q,0.75,interpolation='midpoint')
    q4=np.array(q3_x-q1_x)
    q_total.append(q4)
#Print value of interquartile range
q_total

q_avg=sum(q_total)/len(q_total)

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

#Normalization data
s_norm=np.array(normalize(s_total))
q_norm=np.array(normalize(q_total))

z_norm=np.array(z_norm)
f_norm=np.array(f_norm)

#Combine data together as a feature matrix
matrix=np.append(z_norm,f_norm,axis=1)

matrix1=np.vstack((s_norm,q_norm))

matrix1=matrix1.T

feature_matrix=np.concatenate((matrix,matrix1),axis=1)

#Print feature matrix
feature_matrix

feature_matrix.shape

#Create a feature matrix where each row is a collection of features from each time series.
pca=PCA(n_components=13)

principalComponents=pca.fit_transform(feature_matrix)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2','principal component 3',
                           'principal component 4','principal component 5','principal component 6','principal component 7',
                          'principal component 8','principal component 9','principal component 10','principal component 11'
                         ,'principal component 12','principal component 13'])

#Show the PCA feature matrix with 13 principal components
principalDf

print(pca.explained_variance_ratio_)

# Shows the components value to select top 5 features.
values=abs(pca.components_)
        
#Provide this feature matrix to PCA and derive the new feature matrix. Chose the top 5 features and plot them for each time series. 
values

#Plot 2nd top feature skewness with each time series
for i in range(0,186):
    sk_x=np.linspace(0,186,186)
    sk_y=s_total
    plt.figure()
    plt.plot(sk_x,sk_y)
    plt.show()

#Plot top 1 Feature FFT with each time series
for i in range(0,186):
    fft_x=np.linspace(0,30,30)
    fft_y=final.iloc[i]
    fft_yy=abs(fft(fft_y))
    plt.figure()
    plt.plot(fft_x,fft_yy)
    plt.show()

