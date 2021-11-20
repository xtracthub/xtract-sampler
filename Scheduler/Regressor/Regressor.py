import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso
from xgboost.sklearn import XGBRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

'''
extractor_file: a path to an extractor file with the list of extractors to be analyzed/trained on (simple text with strings
representing extractors)

for example

keyword
image
JSON/XML


data_file: a path to a csv file that corresponds to a file and its known statistics 

for example 

file_path 		extraction time			extracted metadata file size

../pathto/file 1			2s						2b

../pathto/file 2 			4s						4b

...

etc.

returns a pkled dictionary where each key is an extractor and then the value is another 
dictionary where the keys are the metric name and the values are the model 

'''
def generateRegressors(data_file, extractor_file):
	model_pile = dict()
	data_df = pd.read_csv(data_file)
	file_sizes = getFileSizes(data_df['file_path'])
	data_df.insert(1, "file_size", file_sizes)


	with open(extractor_file, "r") as filestream:
		for line in filestream: # assume each extractor is on a seperate line
			metrics = list(data_df.columns.values)[1:] 
			model_pile[line.strip()] = dict.fromkeys(metrics,None) #assume filename is first col
			for metric in metrics:
				pipelines = []
				pipelines.append(('ScaledLR',	 Pipeline([('Scaler', StandardScaler()),('LR',LinearRegression())])))
				pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))
				pipelines.append(('ScaledXGB', 	 Pipeline([('Scaler', StandardScaler()),('EN', XGBRegressor())])))
				pipelines.append(('ScaledKR', 	 Pipeline([('Scaler', StandardScaler()),('KNN', KernelRidge())])))
				pipelines.append(('ScaledSVR', 	 Pipeline([('Scaler', StandardScaler()),('CART', SVR())])))
				pipelines.append(('ScaledGBM', 	 Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())])))


def getFileSizes(files):
	file_sizes = []
	for file in files:
		file_sizes.append(os.path.getsize(file))
	return file_sizes
		

	




generateRegressors("test_data_file.csv", "test_extractor_file.txt")








