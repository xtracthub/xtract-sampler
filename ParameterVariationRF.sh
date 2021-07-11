#!/bin/sh

for n_estimators in 30 100 1000
do 
	for criterion in gini entropy
	do
		for max_depth in 1000 4000 10000
		do
			for min_sample_split in 3 30 300 3000
			do
				python xtract_sampler_main.py --classifier=rf --feature=head --label_csv=../MetaDataExtractionModel/CDIACFileData/labels/cdiac_naivetruth_processed.csv --n_estimators=$n_estimators --criterion=$criterion --max_depth=$max_depth --min_sample_split=$min_sample_split --n=10
			done
		done
	done
done