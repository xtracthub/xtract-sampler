import os
import pickle
from math import log
import numpy as np
import json

class Scheduler:

	def __init__(self, models_dir):
		self.model_piles = dict()
		for subdir, dirs, files in os.walk(models_dir):
			for file_name in files:		
				extractor_name = file_name.split("_")[2].split(".")[0]
				file_path = os.path.join(subdir, file_name)
				with open(file_path, "rb") as f:
					self.model_piles[extractor_name] = pickle.load(f)


	def run(self, prob_vectors, file_sizes):
		priority_list = []


		with open(prob_vectors, "r") as f:
			data = json.load(f)
			for name, vector in data.items():
				for extractor, probability in vector.items():

					time_model = self.model_piles[extractor]['extraction_time']
					if isinstance(time_model, float):
						expected_time = time_model
					else:
						expected_time = time_model.predict(file_sizes[name])
					
					size_model = self.model_piles[extractor]['extraction_size']

					if isinstance(size_model, float):
						expected_size = size_model
					else:
						expected_size = size_model.predict(file_sizes[name])
			
					priority_value = self.calculate_benefit(self, probability, expected_time, expected_size)

					priority_list.append((name, extractor, priority_value))

		return priority_list
	
	# Was called calculating the cost but now we want to frame it as calculating the benefit
	# our objective function
	def calculate_benefit(self, probability, expected_time, expected_size):
		sizes_probability = expected_size * probability + 1 # Laplacian Smoothing
		benefit_raw = sizes_probability / (expected_time + np.finfo(float).eps + 1) # so we don't divide by zero
		return -1 * log(benefit_raw) # priority sorts in increasing order so we flip it around 


if __name__ == "__main__":


	scheduler = Scheduler(os.path.abspath("cdiac_model_piles/"))



	with open("cdiac_probability_predictions.json") as f:
		data = json.load(f)
		for name, value in data.items():
			print(name)
			for key, value in value['probabilities'].items():
				print(key, value)
			break