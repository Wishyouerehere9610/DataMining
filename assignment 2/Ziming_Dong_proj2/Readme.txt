# CSE-572 Project2

For this assignment, we need to develop two code file in Python:

1.train.py
  This python file is for training the pickle machine
 	1.Save the data from meal and no meal csv files.
	2.Clean the data.
	3.Extract features of meal and no meal data
	4.Create label set, meal for 1 and nominal for 0
	5.Combine feature matrix and label set.
	6.Random Shuffle matrix and set up train set and test set for SVM 	machine.
	7.Use k fold cross validation to train data and get accuracy of 	predict result.

2.test.py(execution instruction)
  This python file can take CSV file as input and return predicted labels for the given time series data:
	1.def CSVlist(filename)can be used for test by inputing csvfile.
	if you want to input a new csv file to test, please input the csv name here: data=CSVlist("xxxx.csv). For this python file I just simply use mealData1.csv for test the output function, you are welcome to change the file name as you need. This is the only place you need to change.

	2.Extract the feature from the inputing csv file data

	3.Use the model which I trained from train.py to return the label 0 or 1 for data.

	4.Generate a csv file to restore the label as one column.