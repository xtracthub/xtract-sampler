import sys, os 
sys.path.append(os.path.abspath(".."))
from classifiers import predict
import pickle, json, heapq 
import numpy as np
import pickle as pkl

class Scheduler:
	def __init__(self, class_table_path, sampler_model_file_path, time_model_directory):
		print(os.path.getsize(model_file_path))
		with open(sampler_model_file_path, "rb") as fp1:
			self.model = pickle.load(fp1)
		self.class_table = class_table_path
		self.time_models = get_time_models(time_model_directory)

	def run(self, directory_path):
		index = 0 		
		file_list = []
		for subdir, dirs, files in os.walk(directory_path):
			for file in files:
				filename = os.path.join(subdir, file)
				label, probabilities, file_vector = predict.predict_single_file(filename, self.model, self.class_table, "head")
				file_list.append(file_probability(filename, probabilities))
				index += 1

				if index >= 3:
					break
			if index >= 5:
				break
		

		heapq.heapify(file_list)
		return file_list
	
	def get_time_models(time_model_directory):
		models = dict()
		for subdir, dirs, files in os.walk(time_model_directory):
			for file in files:
				filepath = os.path.join(subdir, file)
				with open(filepath, "rb") as fp:
					pipeline = pkl.load(fp)
					type = file.split("-")[0]
					models[type] = pipeline
		return models

				
class file_probability:
	def __init__(self, file_name, probabilities):
		self.file_name = file_name
		self.probabilities = probabilities
		print(probabilities)
	def __repr__(self):
		return "File path: " + self.file_name + " Probability: " + str(max(self.probabilities.values()))

	def __lt__(self, other):
		return max(self.probabilities.values()) > max(other.probabilities.values())


if __name__ == "__main__":
	scheduler = Scheduler(
	 os.path.abspath("../stored_models/class_tables/rf/CLASS_TABLE-rf-head-2021-07-16-23:20:14.json"),
	 os.path.abspath("../stored_models/trained_classifiers/rf/rf-head-2021-07-16-23:20:14.pkl"))

	queue = scheduler.run("../../CDIACPub8")




	