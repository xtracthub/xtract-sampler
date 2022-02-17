import os
import pickle
from math import log
import numpy as np
import json
import csv

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
		error_files = set()

		file_sizes_dict = self.parse_file_sizes(file_sizes)

		with open(prob_vectors, "r") as f:
			data = json.load(f)
			count = 0
			pairs_processed = 0
			for name, vector in data.items():
				if count % 8000 == 0:
					print("Curr count:", count)

				extrac_count = 0
				for extractor, probability in vector['probabilities'].items():
					if extractor in self.model_piles:
						expected_size = None
						expected_time = None

					
						time_model = self.model_piles[extractor]['extraction_time']
						if isinstance(time_model, float):
							expected_time = time_model
						else:
							try:
								expected_time = time_model.predict(file_sizes_dict[name])[0]
							except KeyError as e:
								print(e)
								error_files.add((name, count))
						size_model = self.model_piles[extractor]['extraction_size']

						if isinstance(size_model, float):
							expected_size = size_model
						else:
							expected_size = size_model.predict(file_sizes_dict[name])[0]

						if expected_time != None and expected_size != None:
							priority_value = self.calculate_benefit(probability, expected_time, expected_size)
							priority_list.append((name, extractor, priority_value))
							pairs_processed += 1

							extrac_count += 1

				count += 1
		print("Files processed: ", count)
		print("Pairs Processed:", pairs_processed)

		return priority_list, list(error_files)
	
	# Was called calculating the cost but now we want to frame it as calculating the benefit
	# our objective function
	def calculate_benefit(self, probability, expected_time, expected_size):
		sizes_probability = expected_size * probability + 1 # Laplacian Smoothing
		benefit_raw = sizes_probability / (expected_time + np.finfo(float).eps + 1) # so we don't divide by zero
		return -1 * log(benefit_raw) # priority sorts in increasing order so we flip it around 

	def parse_file_sizes(self, file_sizes):
		file_sizes_dict = dict()

		with open(file_sizes, "r") as f:
			csv_reader = csv.DictReader(f)
			for row in csv_reader:
				file_sizes_dict[row["path"]] = np.array([row["size"]]).reshape(1, -1)

		return file_sizes_dict
		



if __name__ == "__main__":
	scheduler = Scheduler(os.path.abspath("cdiac_model_piles/"))
	output, error_files = scheduler.run("cdiac_probability_predictions.json", "csv-try-2.csv")

	print("Error files: ", len(error_files))

	with open('cdiac_priority_list_3.pkl', 'wb+') as out:
		pickle.dump(output, out)

	with open('cdiac_error_files_3.pkl', 'wb+') as out:
		pickle.dump(error_files, out)