import sys, os

from numpy.core.fromnumeric import argmax 
sys.path.append(os.path.abspath(".."))
from classifiers import predict
from math import exp, log
import pickle, json, heapq 
import numpy as np
import pickle as pkl
import time
import pandas as pd


class Scheduler:
	def __init__(self, class_table_path, sampler_model_file_path, time_model_directory, size_model_directory, file_crawl_map_path, test=False):
		with open(sampler_model_file_path, "rb") as fp1:
			self.model = pickle.load(fp1)
		self.class_table = class_table_path
		self.time_models = self.get_time_models(time_model_directory)
		self.size_models = self.get_size_models(size_model_directory)
		self.test = test
		self.file_crawl_map = pd.read_csv(file_crawl_map_path)

	def run(self, directory_path):
		index = 0 		
		file_list = []
		heapq.heapify(file_list)
		# we do -1 here because we need to calculate time DIFFERENCES 
		# so we don't need the last element 
		pipeline_times = []

		with open(self.class_table, "r") as fp2:
			class_table = json.load(fp2)

		for i in range(len(self.file_crawl_map.index)):
			file_time = [] # measures 0. filename 1. crawl time 2. feature extraction 3. Prediction Time 4. Heap insertion 5. Extraction time 
			filename = self.file_crawl_map["petrel_path"][i]

			if i == 0:
				elapsed_time = self.file_crawl_map["crawl_timestamp"][i]
			else:
				elapsed_time = self.file_crawl_map["crawl_timestamp"][i] - self.file_crawl_map["crawl_timestamp"][i - 1]
			file_time.append(filename)
			file_time.append(elapsed_time)
			time.sleep(elapsed_time)

			label, probabilities, _, extract_time, predict_time = predict.predict_single_file(filename, self.model, self.class_table, "head")
			
			file_time.append(extract_time)
			file_time.append(predict_time)

			probabilities = np.array(list(probabilities.values())) # sometimes the probabilities are 0
			sizes = self.calculate_estimated_size(filename, class_table)
			times = 1/self.calculate_times(filename, class_table) 
			file_cost = file_estimated_cost(filename, probabilities, sizes, times)
			insert_start_time = time.time()
			heapq.heappush(file_list, file_cost) # TODO: compare heap insertion vs. heapifying everything at the end
			insert_time = time.time() - insert_start_time

			file_time.append(insert_time)
			
			if index % 2000 == 0:
				print("Done with another two thousand:", index)
			
			index += 1

			pipeline_times.append(file_time)

			#if self.test and index >= 6:
			#merely for testing
			#	break

		pipeline_times = pd.DataFrame(pipeline_times, columns=["filename", "crawl_time", "feature_extract_time", "predict_time", "heap_insert_time", "metadata_extract_time"])
		return file_list, pipeline_times
	
	def calculate_times(self, filename, class_table):
		'''
		Note to self we could use these times and probabilities for a dot/cross product?
		'''

		if os.path.getsize(filename) <= 0:
			filesize = 2
		else:
			filesize = os.path.getsize(filename)

		filesize = np.array([filesize]).reshape(1, -1)
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

		for i in range(len(times)):
			if times[i] <= 0:
				times[i] = np.finfo(float).eps

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

	def get_size_models(self, size_model_directory):
		models = dict()
		for subdir, dirs, files, in os.walk(size_model_directory):
			for file in files:
				filepath = os.path.join(subdir, file)
				with open(filepath, "rb") as fp:
					pipeline = pkl.load(fp)
					type = file.split("-")[0]
					models[type.lower()] = pipeline
		models["unknown"] = 0
		models["netcdf"] = 5506.276923076923

		return models

	def calculate_estimated_size(self, filename, class_table):

		if os.path.getsize(filename) <= 0:
			filesize = 2
		else:
			filesize = os.path.getsize(filename)

		filesize = np.array([filesize]).reshape(1, -1)
		sizes = np.zeros(len(class_table.keys()))
		for idx, key in enumerate(class_table.keys()):
			if key == "unknown" or key == "netcdf":
				sizes[idx] = self.size_models[key]
			elif key == "json/xml":
				key = "jsonxml" 
				sizes[idx] = np.exp(self.size_models[key].predict(np.log(filesize)))
			else:
				sizes[idx] = np.exp(self.size_models[key].predict(np.log(filesize)))
		for i in range(len(sizes)):
			if sizes[i] <= 0:
				sizes[i] = np.finfo(float).eps

		return sizes

class file_estimated_cost:
	def __init__(self, file_name, probabilities, sizes, times):
		self.file_name = file_name
		self.probabilities = probabilities
		self.sizes = sizes
		self.times = times
		self.costs = -1 * np.log(np.multiply(self.sizes, 
							np.multiply(self.probabilities, self.times)) 
							+ np.finfo(float).eps)

	def __repr__(self):
		return "File path: " + self.file_name + " Cost: " + str(self.best_extractor())
	def __lt__(self, other):
		return self.best_extractor() > other.best_extractor()
	def best_extractor(self):
		return np.amax(-1 * self.costs)
	def best_extractor_index(self):
		return np.argmax(-1 * self.costs)
	def get_probabilities(self):
		return self.probabilities
	def get_sizes(self):
		return self.sizes
	def get_times(self):
		return self.times
	def get_filename(self):
		return self.file_name

if __name__ == "__main__":
	scheduler = Scheduler(
	 os.path.abspath("../stored_models/class_tables/rf/CLASS_TABLE-rf-head-2021-07-22-16:47:16.json"),
	 os.path.abspath("../stored_models/trained_classifiers/rf/rf-head-2021-07-22-16:47:16.pkl"),
	 os.path.abspath("EstimateTime/models"), os.path.abspath("EstimateSize/models"),
	 "filename_crawl_t_map_processed.csv", False)

	start_time = time.time()
	queue, times = scheduler.run("../../CDIACPub8")
	print("--- %s seconds ---" % (time.time() - start_time))
	print(times.head())


	times.to_csv("times.csv")
	with open("queue.pkl", "wb+") as fp:
		pkl.dump(queue, fp)


	'''
	for elem in queue:
		print(elem)
	'''
