import sys, os 
sys.path.append(os.path.abspath(".."))
from classifiers import predict
from math import exp, log
import pickle, json, heapq 
import numpy as np
import pickle as pkl


class Scheduler:
	def __init__(self, class_table_path, sampler_model_file_path, time_model_directory):
		with open(sampler_model_file_path, "rb") as fp1:
			self.model = pickle.load(fp1)
		self.class_table = class_table_path
		self.time_models = self.get_time_models(time_model_directory)

	def run(self, directory_path):
		index = 0 		
		file_list = []
		for subdir, dirs, files in os.walk(directory_path):
			for file in files:
				filename = os.path.join(subdir, file)
				label, probabilities, _ = predict.predict_single_file(filename, self.model, self.class_table, "head")
				probabilities = ((1/(np.array(list(probabilities.values()))  + np.finfo(float).eps ))) # sometimes the probabilities are 0
				times = np.exp(self.calculate_times(filename))
				costs = np.multiply(probabilities, times)

				file_list.append(file_estimated_cost(filename, costs))
				index += 1

				if index >= 2:
					break
			if index >= 10:
				break

		heapq.heapify(file_list)
		return file_list
	
	def calculate_times(self, filename):
		'''
		Note to self we could use these times and probabilities for a dot/cross product?
		'''
		with open(self.class_table, "r") as fp2:
			class_table = json.load(fp2)
		filesize = np.array([os.path.getsize(filename)]).reshape(1, -1)
		times = np.zeros(len(class_table.keys()))
		for idx, key in enumerate(class_table.keys()):
			if key == "unknown":
				times[idx] = 0.5
			elif key == "keyword":
				times[idx] = np.exp(self.time_models[key].predict(np.log(filesize)))
			elif key == "netcdf":
				times[idx] = self.time_models[key].predict(np.log(filesize))
			elif key == "json/xml":
				key = "jsonxml"
				times[idx] = self.time_models[key].predict(filesize)
			else:
				times[idx] = self.time_models[key].predict(filesize)
		return times

	def get_time_models(self, time_model_directory):
		models = dict()
		for subdir, dirs, files in os.walk(time_model_directory):
			for file in files:
				filepath = os.path.join(subdir, file)
				with open(filepath, "rb") as fp:
					pipeline = pkl.load(fp)
					type = file.split("-")[0]
					models[type.lower()] = pipeline
		models["unknown"] = 0.5
		return models

class file_estimated_cost:
	def __init__(self, file_name, costs):
		self.file_name = file_name
		self.costs = costs
	def __repr__(self):
		return "File path: " + self.file_name + " Cost " + str(np.amin(self.costs))
	def __lt__(self, other):
		return np.amin(self.costs) < np.amin(other.costs)


if __name__ == "__main__":
	scheduler = Scheduler(
	 os.path.abspath("../stored_models/class_tables/rf/CLASS_TABLE-rf-head-2021-07-16-23:20:14.json"),
	 os.path.abspath("../stored_models/trained_classifiers/rf/rf-head-2021-07-16-23:20:14.pkl"),
	 os.path.abspath("EstimateTime/models"))

	queue = scheduler.run("../../CDIACPub8")

	for elem in queue:
		print(elem)




	
