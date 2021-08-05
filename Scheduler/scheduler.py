from re import L
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
from multiprocessing.managers import BaseManager, MakeProxyType, public_methods, SyncManager
from queue import PriorityQueue, Empty
from ctypes import c_char_p

class ProxyPriorityQueue(PriorityQueue):
	def get_attribute(self, name):
		return getattr(self, name)

class CustomManager(SyncManager):
	pass


class file_extractor_estimated_cost:
	def __init__(self, file_name, extractor, probability, size, time):
		self.file_name = file_name
		self.probability = probability
		self.size = size
		self.time = time
		self.cost = -1 * self.calculate_cost()
		self.extractor = extractor
	def get_cost(self):
		return self.cost
	def get_probability(self):
		return self.probability
	def get_sizes(self):
		return self.size
	def get_times(self):
		return self.time
	def get_filename(self):
		return self.file_name
	def get_extractor(self):
		return self.extractor
	def __repr__(self):
		return "File path: " + self.file_name + "Extractor: " + self.extractor + " Cost: " + str(self.cost)
	def __lt__(self, other):
		return self.cost > other.cost
	def calculate_cost(self):
		sizes_probability = self.size * self.probability + 1  # smoothing
		cost_raw = sizes_probability / (self.time + np.finfo(float).eps + 1) # so we don't get divide by zero cost
		return -1 * log(cost_raw)

'''
	DO NOT USE 
	LEGACY CLASS
class file_estimated_cost:
	def __init__(self, file_name, probabilities, sizes, times):
		self.file_name = file_name
		self.probabilities = probabilities
		self.sizes = sizes
		self.times = times
		#self.costs = -1 * self.probabilities
		self.costs = self.calculate_cost()

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
	def calculate_cost(self):
		sizes_probabilities = np.multiply(self.sizes, self.probabilities) + 1  # smoothing
		cost_raw = np.multiply(sizes_probabilities, self.times) + np.finfo(float).eps # so we get nonzero cost
		return -1 * np.log(cost_raw)


class ProxyFileEstimatedCost(file_estimated_cost):
	def get_attribute(self, name):
		return getattr(self, name)
'''

class Scheduler:
	def __init__(self, class_table_path, sampler_model_file_path, time_model_directory, size_model_directory, file_crawl_map_path, actual_extraction_times, thresholds, test=False):
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

		self.file_count_threshold = (len(self.file_crawl_map.index) * thresholds * 5).astype(int)

		self.manager = self.get_manager()
		self.crawl_queue = self.manager.PriorityQueue()
		self.xtract_queue = self.manager.PriorityQueue()
		self.dequeue_list = self.manager.Queue()
		self.file_index = mp.Value('i', 0)
		self.zero_extraction = mp.Value('i', 0)

		self.start_time = None

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
		#FileEstimatedCostProxy = MakeProxyType("file_estimated_cost", public_methods(file_estimated_cost))

		CustomManager.register("PriorityQueue", PriorityQueue, PriorityQueueProxy)
		#CustomManager.register("file_estimated_cost", file_estimated_cost, FileEstimatedCostProxy)
		m = CustomManager()
		m.start()
		return m

	def run(self):
	
		lock = mp.Lock()

		if mp.cpu_count() % 2 != 0:
			print("This program only works on even-cored processors")
			exit()

		
		enqueue_processes=[mp.Process(target=self.enqueue, args=(lock,)) for x in range(0, int(mp.cpu_count() / 2))]
		dequeue_processes=[mp.Process(target=self.dequeue, args=(lock, self.file_index)) for x in range(0, int(mp.cpu_count() / 2))]

		print("Extraction threshold: ", self.file_count_threshold)

		self.simulate_crawl()

		self.start_time = time.time()

		for p in enqueue_processes:
			p.start()
		
		for p in dequeue_processes:
			p.start()
		
		for p in dequeue_processes:
			p.join()

		for p in enqueue_processes:
			p.join()

		scheduler_run_time = time.time() - self.start_time
		print("--- %s seconds ---" % (scheduler_run_time))
		print("Zero Extraction Times", self.zero_extraction.value)
		print("Length of dict:", self.dequeue_list.qsize())

		dequeue_list = []
		while not self.dequeue_list.empty():
				dequeue_list.append(self.dequeue_list.get())

		with open('Experiment4/dequeue_list.pkl', 'wb+') as output:
				pkl.dump(dequeue_list, output)
		
		return scheduler_run_time 
		#with open("Experiment2/dequeue_list_threshold_{th}.json".format(th=self.extraction_threshold), "w+") as fp:
		#	json.dump(dequeue_dict, fp, indent=4)

	def enqueue(self, lock):
		while not self.crawl_queue.empty():
			#print("Enqueue file tuple lock")
			#lock.acquire()
			try:
				file_tuple = self.crawl_queue.get(timeout=120)
			except Empty:
				#print("Nothing in crawl queue")
				pass
			#finally:
			#	print("Enqueue file tuple lock released")
			#	lock.release()
			_, file_path = file_tuple
			file_costs = self.calculate_costs(file_path)
			#print("Enqueue push onto the heap lock")
			#lock.acquire()
			#try:
			for file_cost in file_costs:
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
				#print("Nothing in xtract queue")
				pass	
			except TypeError:
				print(type(self.xtract_queue))
				exit()
			except Empty:
				#print("Took too long to get from queue")
				pass
			#finally:
				#lock.release()
			if file_cost != None:
				extractor = file_cost.get_extractor()
				extractor_idx = self.class_table_dict[extractor] - 1
				if extractor == 'unknown': #skip unknowns
					continue
				extraction_time = self.extraction_times.loc[file_cost.get_filename()][extractor_idx]
				time.sleep(extraction_time)
				file_index.value += 1

				if file_index.value in self.file_count_threshold:
					print("----- Dequeue: {f} files Time: {t} -----".format(f=file_index.value, t=time.time() - self.start_time))

				if extraction_time == 0:
					self.zero_extraction.value += 1
				self.dequeue_list.put((file_cost.get_filename(), file_cost.get_extractor(), file_cost.get_cost()))
				#self.dequeue_list[self.manager.Value(c_char_p, file_cost.get_filename())] = self.manager.Value('i', best_index)
				#print("Dequeue:", file_index.value, "Extraction time:", extraction_time)


	def calculate_costs(self, filename):
		file_list = []
		label, probabilities, _, extract_time, predict_time = predict.predict_single_file(filename, self.model, self.class_table, "head")
		probabilities = self.convert_probabilities_to_dict(np.array(list(probabilities.values()))) # sometimes the probabilities are 0
		sizes = self.calculate_estimated_size(filename, self.class_table_dict)
		times = self.calculate_times(filename, self.class_table_dict)

		for key in self.class_table_dict:
			file_list.append(file_extractor_estimated_cost(filename, key, probabilities[key], sizes[key], times[key]))

		return file_list

	def convert_probabilities_to_dict(self, probabilities):
		probabilities_dict = dict()
		for idx, key in enumerate(self.class_table_dict):
			probabilities_dict[key] = probabilities[idx]
		return probabilities_dict 	

	def calculate_times(self, filename, class_table):
		'''
		Note to self we could use these times and probabilities for a dot/cross product?
		'''

		if os.path.getsize(filename) <= 0:
			filesize = 2
		else:
			filesize = os.path.getsize(filename)

		filesize = np.array([filesize]).reshape(1, -1)
		times = dict()
		for key in class_table.keys():
			if key == "unknown":
				times[key] = 0.5
			elif key == "keyword":
				times[key] = np.exp(self.time_models[key].predict(np.log(filesize)))
			elif key == "netcdf":
				times[key] = self.time_models[key].predict(np.log(filesize))
			elif key == "json/xml":
				times[key] = self.time_models["jsonxml"].predict(filesize)
			else:
				times[key] = self.time_models[key].predict(filesize)

		for key, value in times.items():
			if value <= 0:
				value = np.finfo(float).eps

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
		models["unknown"] = 100
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
		sizes = dict()
		for key in class_table.keys():
			if key == "unknown" or key == "netcdf":
				sizes[key] = self.size_models[key]
			elif key == "json/xml":
				sizes[key] = np.exp(self.size_models["jsonxml"].predict(np.log(filesize)))
			else:
				sizes[key] = np.exp(self.size_models[key].predict(np.log(filesize)))
		
		for key, value in sizes.items():
			if value <= 0:
				value = np.finfo(float).eps

		return sizes
	
	def get_crawl_queue(self):
		return self.crawl_queue


def run_experiments(input_thresholds):
		scheduler = Scheduler(
		os.path.abspath("../stored_models/class_tables/rf/CLASS_TABLE-rf-head-2021-07-22-16:47:16.json"),
		os.path.abspath("../stored_models/trained_classifiers/rf/rf-head-2021-07-22-16:47:16.pkl"),
		os.path.abspath("EstimateTime/models"), os.path.abspath("EstimateSize/models"),
		"filename_crawl_t_map_processed.csv", "AggregateExtractionTimes/ExtractionTimes.csv", thresholds=input_thresholds, test=False)
		times = scheduler.run()
		return times


if __name__ == "__main__":
	# Threshold of .0005 is for 10 files

	#thresholds = [.0005]
	thresholds = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
	#time_output = open('Experiment3/Times.txt', 'w+') 
		
	print("-----------------------------------------------------")
	run_time = run_experiments(thresholds)
	#time_output.write('Total Time in seconds: ' + str(run_time) + '\n')
	print("-----------------------------------------------------")

	#time_output.close()
	
	#print(times.head())


	#times.to_csv("times.csv")
	#with open("queue.pkl", "wb+") as fp:
	#pkl.dump(queue, fp)


	'''
	for elem in queue:
		print(elem)
	'''
