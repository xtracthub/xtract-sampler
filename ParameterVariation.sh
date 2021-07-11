#!/bin/sh

for iter in 100 500 1000
do
	for C in 0.5 1 5
	do
		for kernel in linear rbf sigmoid 
		do
			python xtract_sampler_main.py --classifier=svc --feature=head --label_csv=../MetaDataExtractionModel/CDIACFileData/labels/cdiac_naivetruth_processed.csv --C=$C --iter=$iter --kernel=$kernel --n=10
		done
		for degree in 3 5 10
		do
			python xtract_sampler_main.py --classifier=svc --feature=head --label_csv=../MetaDataExtractionModel/CDIACFileData/labels/cdiac_naivetruth_processed.csv --C=$C --iter=$iter --kernel=$kernel --degree=$degree --n=10
		done 

		for penalty in none l2
		do
			for solver in lbfgs sag newton-cg liblinear
			do
				python xtract_sampler_main.py --classifier=logit --feature=head --label_csv=../MetaDataExtractionModel/CDIACFileData/labels/cdiac_naivetruth_processed.csv --C=$C --iter=$iter --penalty=$penalty --solver=$solver --n=10
			done
		done

		for solver in liblinear saga
		do
			python xtract_sampler_main.py --classifier=logit --feature=head --label_csv=../MetaDataExtractionModel/CDIACFileData/labels/cdiac_naivetruth_processed.csv --C=$C --iter=$iter --penalty=l1 --solver=$solver --n=10
		done

	done
done 