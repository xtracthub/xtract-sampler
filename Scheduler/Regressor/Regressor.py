import numpy as np
import pandas as pd
import os
import pickle as pkl

from canova_source import canova

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso
from xgboost.sklearn import XGBRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score

from scipy.stats import pearsonr


'''
extractor: the extractor associated with the data file 
all data points in the data_file are assumed to be from this extractor

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
def generateRegressors(data_file, extractor):

	data_df = pd.read_csv(data_file)

	metrics = list(data_df.columns.values)[1:] 
	model_pile = dict.fromkeys(metrics,None) #assume filename is first col
	
	file_sizes = getFileSizes(data_df['file_path'])
	data_df.insert(1, "file_size", file_sizes, allow_duplicates=False)
	
	
	print(metrics)
	
	for metric in metrics:
		linear_pipelines = dict()
		nonlinear_pipelines = dict()

		linear_pipelines['ScaledLR'] =	Pipeline([('Scaler', StandardScaler()),('LR',LinearRegression())])
		linear_pipelines['ScaledLASSO'] = Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])
		linear_pipelines['ScaledSVR'] =	 Pipeline([('Scaler', StandardScaler()),('CART', SVR())])

		nonlinear_pipelines['ScaledXGB'] = Pipeline([('Scaler', StandardScaler()),('EN', XGBRegressor())])
		nonlinear_pipelines['ScaledKR'] = 	 Pipeline([('Scaler', StandardScaler()),('KNN', KernelRidge())])
		nonlinear_pipelines['ScaledGBM'] =	 Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())])

		X = data_df['file_size'].to_numpy()
		Y = data_df[metric].to_numpy()

		r, _ = pearsonr(X, Y) # disregard p-value
		canova_value = canova(X, Y)
		
		X = X.reshape(-1, 1)

		linear_scores = dict()
		nonlinear_scores = dict()
		
		if r >= 0.4:
			# correlated
			for name, model in linear_pipelines.items():
				kfold = KFold(n_splits = 2, shuffle=True) # Split into ten folds and hope the bias isn't too high (there aren't TOO many files)
				r2_score = np.mean(cross_val_score(model, X, Y, cv=kfold, scoring='r2'))
				
				if r2_score > 0:
					linear_scores[name] = r2_score  		
					print("r2_score: ", r2_score)

		if canova_value >= 0.4:
			for name, model in nonlinear_pipelines.items():
				kfold = KFold(n_splits = 2, shuffle=True) # Split into ten folds and hope the bias isn't too high (there aren't TOO many files)
				r2_score = np.mean(cross_val_score(model, X, Y, cv=kfold, scoring='r2'))

				if r2_score > 0:
					nonlinear_scores[name] = r2_score
					print("r2_score: ", r2_score)	
	
		if len(linear_scores) == 0 and len(nonlinear_scores) == 0: # in cases all regressions are bad
			model_pile[metric] = data_df[metric].mean()
		else:
			model_pile[metric] = pickBestModel(linear_scores, nonlinear_scores, X, Y, linear_pipelines, nonlinear_pipelines)

	pkl.dump(model_pile, open("Model_Pile_" + extractor + ".pkl", "wb"))
	return model_pile

def pickBestModel(linear_scores, nonlinear_scores, X, Y, linear_pipelines, nonlinear_pipelines):
	max_score = 0
	model = None
	isLinearModel = None
	for name, score in linear_scores.items():
		if score > max_score:
			max_score = score
			model = name
			isLinearModel = True
	for name, score in nonlinear_scores.items():
		if score > max_score:
			max_score = score
			model = name
			isLinearModel = False

	assert model != None
	assert isLinearModel != None

	print("Model: ", model)
	print("Score: ", max_score)
	if isLinearModel:
		best_model = linear_pipelines[model]
	else:
		best_model = nonlinear_pipelines[model]
	best_model.fit(X, Y)
	return best_model

def getFileSizes(files):
	file_sizes = []
	for file in files:
		file_sizes.append(os.path.getsize(file))
	return file_sizes
		
model_pile = generateRegressors("test_data_file.csv", "keyword")
print(model_pile)








