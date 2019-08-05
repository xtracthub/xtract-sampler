FROM python:latest
 
MAINTAINER Ryan Wong

COPY classify.py extpredict.py feature.py headbytes.py main.py predict.py randbytes.py randhead.py test_model.py train_model.py rf-head-default.pkl naivetruth.csv /

RUN pip install sklearn numpy

ENTRYPOINT ["python", "main.py"]
