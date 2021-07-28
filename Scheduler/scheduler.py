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
import multiprocessing as mp
from multiprocessing.managers import BaseManager, MakeProxyType, public_methods
from queue import PriorityQueue, Empty

class ProxyPriorityQueue(PriorityQueue):
	def get_attribute(self, name):
		return getattr(self, name)


class CustomManager(BaseManager):
	pass

class file_estimated_cost:
	def __init__(self, file_name, probabilities, sizes, times):
		self.file_name = file_name
		self.probabilities = probabilities
		self.sizes = sizes
		self.times = times
		self.costs = -1 * np.log(np.multiply(self.sizes, 
							np.multiply(self.probabilities, self.times)) 
							+ np.finfo(float).eps)
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
	def __repr__(self):
		return "File path: " + self.file_name + " Cost: " + str(self.best_extractor())
	def __lt__(self, other):
		return self.best_extractor() > other.best_extractor()

class ProxyFileEstimatedCost(file_estimated_cost):
	def get_attribute(self, name):
		return getattr(self, name)


class Scheduler:
	def __init__(self, class_table_path, sampler_model_file_path, time_model_directory, size_model_directory, file_crawl_map_path, actual_extraction_times, test=False):
		with open(sampler_model_file_path, "rb") as fp1:
			self.model = pickle.load(fp1)
		self.class_table = class_table_path
		
		with open(self.class_table, "r") as fp2: 
			self.class_table_dict = json.load(fp2)
		
		self.extraction_times = pd.read_csv(actual_extraction_times, index_col=0)

		self.time_models = self.get_time_models(time_model_directory)
		self.size_models = self.get_size_models(size_model_directory)
		self.test = test
		self.file_crawl_map = pd.read_csv(file_crawl_map_path)
		
		self.manager = self.get_manager()
		self.crawl_queue = self.manager.PriorityQueue()
		self.xtract_queue = self.manager.PriorityQueue()
		self.file_index = mp.Value('i', 0)

	def simulate_crawl(self):
		for i in range(len(self.file_crawl_map.index)):
			self.crawl_queue.put((self.file_crawl_map["crawl_timestamp"][i], self.file_crawl_map["petrel_path"][i]))
			if i == 0:
				elapsed_time = self.file_crawl_map["crawl_timestamp"][i]
			else:
				elapsed_time = self.file_crawl_map["crawl_timestamp"][i] - self.file_crawl_map["crawl_timestamp"][i - 1]
			time.sleep(elapsed_time)


	def get_manager(self):
		PriorityQueueProxy = MakeProxyType("PriorityQueue", public_methods(PriorityQueue))
		FileEstimatedCostProxy = MakeProxyType("file_estimated_cost", public_methods(file_estimated_cost))

		CustomManager.register("PriorityQueue", PriorityQueue, PriorityQueueProxy)
		#CustomManager.register("file_estimated_cost", file_estimated_cost, FileEstimatedCostProxy)
		m = CustomManager()
		m.start()
		return m

	def run(self, directory_path):
		start_time = time.time()
		lock = mp.Lock()

		if mp.cpu_count() % 2 != 0:
			print("This program only works on even-cored processors")
			exit()

		
		enqueue_processes=[mp.Process(target=self.enqueue, args=(lock,)) for x in range(0, int(mp.cpu_count() / 2))]
		dequeue_processes=[mp.Process(target=self.dequeue, args=(lock, self.file_index)) for x in range(0, int(mp.cpu_count() / 2))]

		self.simulate_crawl() 

		for p in enqueue_processes:
			p.start()
		
		for p in dequeue_processes:
			p.start()
		
		for p in dequeue_processes:
			p.join()

		for p in enqueue_processes:
			p.join()

	
		print("--- %s seconds ---" % (time.time() - start_time))

	def enqueue(self, lock):
		while not self.crawl_queue.empty():
			#print("Enqueue file tuple lock")
			#lock.acquire()
			#try:
			file_tuple = self.crawl_queue.get()
			#finally:
			#	print("Enqueue file tuple lock released")
			#	lock.release()
			_, file_path = file_tuple
			file_cost = self.calculate_costs(file_path)
			
			#print("Enqueue push onto the heap lock")
			#lock.acquire()
			#try:
			self.xtract_queue.put(file_cost)
			#finally:
			#	print("Enqueue push onto the heap lock released")
			#	lock.release()


	def dequeue(self, lock, file_index):
		while not self.xtract_queue.empty() or not self.crawl_queue.empty():
			#lock.acquire()
			file_cost = None
			try:
				file_cost = self.xtract_queue.get(timeout=120)
			except IndexError:
				print("Nothing in xtract queue")
				pass	
			except Empty:
				print("Took too long to get from queue")
				pass
			#finally:
				#lock.release()
			if file_cost != None:
				best_index = file_cost.best_extractor_index()
				if best_index == 2: #skip unknowns
					continue
				extraction_time = self.extraction_times.loc[file_cost.get_filename()][best_index]
				time.sleep(extraction_time)
				file_index.value += 1
				print("Dequeue:", file_index.value, "Extraction time:", extraction_time)


	def calculate_costs(self, filename):
			label, probabilities, _, extract_time, predict_time = predict.predict_single_file(filename, self.model, self.class_table, "head")
			probabilities = np.array(list(probabilities.values())) # sometimes the probabilities are 0
			sizes = self.calculate_estimated_size(filename, self.class_table_dict)
			times = 1/self.calculate_times(filename, self.class_table_dict) 
			file_cost = file_estimated_cost(filename, probabilities, sizes, times)

			return file_cost

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
		models["unknown"] = 10
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
	
	def get_crawl_queue(self):
		return self.crawl_queue


if __name__ == "__main__":
	scheduler = Scheduler(
	 os.path.abspath("../stored_models/class_tables/rf/CLASS_TABLE-rf-head-2021-07-22-16:47:16.json"),
	 os.path.abspath("../stored_models/trained_classifiers/rf/rf-head-2021-07-22-16:47:16.pkl"),
	 os.path.abspath("EstimateTime/models"), os.path.abspath("EstimateSize/models"),
	 "filename_crawl_t_map_processed.csv", "AggregateExtractionTimes/ExtractionTimes.csv", False)

	start_time = time.time()


	times = scheduler.run("../../CDIACPub8")
	print("--- %s seconds ---" % (time.time() - start_time))
	#print(times.head())


	#times.to_csv("times.csv")
	#with open("queue.pkl", "wb+") as fp:
	#pkl.dump(queue, fp)


	'''
	for elem in queue:
		print(elem)
	'''
