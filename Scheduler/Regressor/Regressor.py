import pandas as pd

'''
extractor_file: a path to an extractor file with the list of extractors to be analyzed/trained on (simple text with strings
representing extractors)

for example

keyword
image
JSON/XML


data_file: a path to a csv file that corresponds to a file and its known statistics 

for example 

file path 		extraction time			extracted metadata file size

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
	print(data_df)
	with open(extractor_file, "r") as filestream:
		for line in filestream: # assume each extractor is on a seperate line
			metrics = list(data_df.columns.values)[1:] 
			model_pile[line.strip()] = dict.fromkeys(metrics,None) #assume filename is first col
			for metric in metrics:



	




generateRegressors("test_data_file.csv", "test_extractor_file.txt")








