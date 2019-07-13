# run experiments
#!/bin/bash

echo "Running rf experiments"
python main.py --classifier rf --feature head
echo "Done"
python main.py --classifier rf --feature rand
echo "Done
python main.py --classifier rf --feature randhead --head-bytes 256 --rand-bytes 256
echo "Done"
python main.py --classifier rf --feature ngram --ngram 1
echo "Done"

echo "Running svc experiments"
python main.py --classifier svc --feature head
python main.py --classifier svc --feature rand
python main.py --classifier svc --feature randhead --split 0.5 --head-bytes 256 --rand-bytes 256
python main.py --classifier svc --feature ngram --split 0.5 --ngram 1

echo "Running logit experiments"
python main.py --classifier logit --feature head
python main.py --classifier logit --feature rand
python main.py --classifier logit --feature randhead --head-bytes 256 --rand-bytes 256
python main.py --classifier logit --feature ngram --ngram 1

