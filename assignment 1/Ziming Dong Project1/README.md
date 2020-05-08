# CSE-572 Project 1
Detailing Requirements

Steps in Python file:
1. Combine 5 csv time series glucose csv file.
2. Clean the missing value.
3. Use four types feature extraction methods to calculate the value for each time series.
4. Combine all features values as a feature matrix.
5. Use PCA to get principal components feature matrix.
6. Get the 5 highest values in components values array.
7. Plot the highest values correspond features for each time series.

Requirements:
a)	Extract 4 different types of time series features from only the CGM data cell array and CGM timestamp cell array.

1.Polynomial Curve Fitting
2.Fast Fourier Transform
3.Skewness
4.Interquartile Range

b)	For each time series explain why you chose such feature.

See the detail in project report.

c)	Show values of each of the features and argue that your intuition in step b is validated or disproved? 

See the value of each of the features in python file and arugment in report.

1.Polynomial Curve Fitting(z_total)
2.Fast Fourier Transform(f_total)
3.Skewness(s_total)
4.Interquartile Range(q_total)

d)	Create a feature matrix where each row is a collection of features from each time series. SO if there are 75 time series and your feature length after concatenation of the 4 types of features is 17 then the feature matrix size will be 75 X 17.

See the printed feature matrix in python file.(feature_matrix)

e)	Provide this feature matrix to PCA and derive the new feature matrix. Chose the top 5 features and plot them for each time series.

See the principalDf in python file and top 5 feature selection in report.

f)	For each feature in the top 5 argue why it is chosen as a top five feature in PCA? 

See the detail explaination in project report.
