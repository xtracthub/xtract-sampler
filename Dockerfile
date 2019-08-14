FROM python:3.6
 
MAINTAINER Ryan Wong

RUN mkdir xtract-sampler/
RUN mkdir xtract-sampler/automated_training_results/


COPY automated_training_results/naivetruth.csv xtract-sampler/automated_training_results/
COPY classify.py extpredict.py feature.py headbytes.py xtract_sampler_main.py \
     predict.py randbytes.py randhead.py test_model.py train_model.py \
     rf-head-default.pkl xtract-sampler/
COPY requirements.txt /

RUN git clone https://github.com/xtracthub/xtract-tabular && git clone https://github.com/xtracthub/xtract-jsonxml \
    && git clone https://github.com/xtracthub/xtract-netcdf && git clone https://github.com/xtracthub/xtract-keyword
RUN pip install -U nltk
RUN pip install -r requirements.txt

#ENTRYPOINT ["python", "main.py"]
