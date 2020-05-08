#CSE 572 Project3

For this assignment,I develop two code file in python:

1.train.py:
	1.This python file is to take mealData1-5 as input and extract feature
 	  for those data.
	2.Extract the ground truth bin_label for each row.
	3.Use K-means and DBscan to do clustering, and return their corresponding
	  label save as clustering result.
	4.Use k-fold validation split data to do KNN to find the label for splited
	  test data and report SSE and Accuracy for k-means and dbscan
	5.Generate three pickle files: feature.pickle, db_label.pickle and km_label.pickle
	  for using in test.py
	6.Please make sure train.py and all data files are under same folder, including 
	  five mealAmount files and five mealData files.
	7.After running the train.py, the accuracy and SSE for k-means and dbscan should 	  be printed in order.
	
2.test.py:
	1.This python file takes input as file name to do the clustering.
	2.This python file can extract input data feature and for each data's
	  feature, use KNN to classified each data's label for both k-means and
	  dbscan.
	3.Create a new csv file named"labels.csv", then add clustering result from k-means
  	  and dbscan to this csv file.
	6.Please make sure input test file, test.py and three pickle files are in the same folder
	7.After running the test.py, a csv file named "labels.csv" should be generated.


How to run the test file:
	1.In test.py, there is a line "Data=CSVlist('test.csv')" you can change test.csv to name of the input file.
	   Please make sure them and pickle files are under same folder, otherwise you will input the whole file path.
	2.After running test.py, there should be a csv file named "labels.csv" generated.
	3.Both k-means and dbscan clustering result are in this csv file.
