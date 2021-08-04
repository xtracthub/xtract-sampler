#!/bin/sh


for i in 1 2 4 8 16 32 64 128 256 512 1024
do 
	python xtract_sampler_main.py --mode train --classifier rf --feature head --label_csv ../MetaDataExtractionModel/CDIACFileData/labels/cdiac_naivetruth_processed.csv --head_bytes=$i
done

echo "Done with RF."

for i in 1 2 4 8 16 32 64 128 256 512 1024
do 
	python xtract_sampler_main.py --mode train --classifier svc --feature head --label_csv ../MetaDataExtractionModel/CDIACFileData/labels/cdiac_naivetruth_processed.csv --head_bytes=$i
done

echo "Done with SVC."

for i in 1 2 4 8 16 32 64 128 256 512 1024
do 
	python xtract_sampler_main.py --mode train --classifier logit --feature head --label_csv ../MetaDataExtractionModel/CDIACFileData/labels/cdiac_naivetruth_processed.csv --head_bytes=$i
done

echo "Done!"