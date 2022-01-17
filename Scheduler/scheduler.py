import os
import pickle
from math import log
import numpy as np

class Scheduler:

	def __init__(self, models_dir):
		self.model_piles = dict()
		for subdir, dirs, files in os.walk(models_dir):
			for file_name in files:		
				extractor_name = file_name.split("_").split(".")[0]
				with open(file_name, "rb") as f:
					self.model_piles[extractor_name] = pickle.load(file_name)


	def run(self, prob_vectors, file_sizes):
		priority_list = []
		for probability_tuple in prob_vectors:
			file_name, extractor, probability = probability_tuple
			expected_time = self.model_piles[extractor]['extraction_time'].predict(file_sizes[file_name])
			expected_size = self.model_piles[extractor]['extraction_size'].predict(file_sizes[file_name])

			priority_value = self.calculate_benefit(self, probability, expected_time, expected_size)

			priority_list.append((file_name, extractor, priority_value))

		return priority_list
	
	# Was called calculating the cost but now we want to frame it as calculating the benefit
	# our objective function
	def calculate_benefit(self, probability, expected_time, expected_size):
		sizes_probability = expected_size * probability + 1 # Laplacian Smoothing
		benefit_raw = sizes_probability / (expected_time + np.finfo(float).eps + 1) # so we don't divide by zero
		return -1 * log(benefit_raw) # priority sorts in increasing order so we flip it around 