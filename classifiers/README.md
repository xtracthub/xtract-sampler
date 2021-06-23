## Code-Walkthrough

This README is meant to describe how both training and prediction calls from the command line (for these commands, see 
/xtract-sampler/README.md). 

## Train
In order to train the model, the user provides a request that resembles the following:
```
python xtract_sampler_main.py --mode train --classifier rf --feature head --label_csv automated_training_results/tyler_truth.csv
```
This means that the training invocation is received in `xtract_sampler_main.py`, whose main method calls the `extract_sampler`
function. In this function the `if mode == 'train':` block creates one of three `feature` objects: 
- HeadBytes (all bytes from the head of the file)
- RandBytes (bytes from throughout the file, at random)
- RandHead (n bytes from head, m bytes from throughout file)

For this example, we will explore HeadBytes(). This class only inputs the number of 
bytes to read from the head, and the class is stored in features/headbytes.py. 
You will notice that there are functions for `get_feature()` and `translate()`, 
but neither is invoked. Now back to `xtract_sampler_main.py`...

Now that we have an empty feature class object, we want to load it. In the line 
that reads
```
reader = NaiveTruthReader(features, labelfile=label_csv)
```
you will input your HeadBytes() feature object, AS WELL AS the path to a file containing ground-truth 
data for each file in our training set (we get this via automated training, which runs each extractor on each
file. We will not go into this in detail here). The code for the NaiveTruthReader() object is located in 
`features/readers/readers.py`. 

The NaiveTruthReader() class object contains two functions: 
- `extract_row_data()` which scans the inputted CSV file, line-by-line. It
then invokes the `get_features()` method, and stores row data data in the following
list format `[full_path, file_name, features, extractor_label]`.
- `run()` scales via the multiprocessing library, and performs `extract_row_data()` by using multiple concurrent workers. 

### But what exactly are the features? 
Let's look at the `get_feature()` method in `features/headbytes.py`. 

This inputs an open file object (opened by the NaiveTruthReader), and in the while-loop
reads up to `head_size` bytes from the start of the file, OR all of the bytes, whichever comes 
first. 

In the case where the total number of bytes in a file is less than the head, bytes, 
we append the remaining space with the 'null' byte. 

```
if len(head) < self.head_size:
    head.extend([b'' for i in range(self.head_size - len(head))])
```

So in the case where `head_size=10`, we will have either of the following byte streams: 
(A) <byte_0>, <byte_1>, ..., <byte_9> (when the file size is greater than the head size)
(B) <byte_0>, <byte_1>, <byte_2>, <null_byte>, <null_byte>, ... (when file size is less than head size)

So now the NaiveTruthReader() is holding these byte-features (as part of the aforementioned list) in `self.data`
If we look back at `xtract_sampler_main.py`, we see that our reader is fed to the experiment() function.
It extracts the aforementioned features via a call to teh `NaiveTruthReader().run()` command, 
You will then see the following line of code in `experiment()`: 
```
classifier = ModelTrainer(reader, class_table_path=class_table_path, classifier=classifier_name, split=split)
```

In `classifiers/train_model.py`, you will find the ModelTrainer() class. In its `__init__()`
function, there are well-documented steps that include splitting the data into train/test cases, 
creating empty feature vectors, and loading said feature vectors with data. 
One key thing to note are these lines: 
``` 
x, y = reader.feature.translate(raw_data[i])
X[i] = x
Y[i] = y
```
Each item in raw_data\[i] looks roughly like the following: 
```
['/Users/tylerskluzacek/Desktop/pub8/oceans/VOS_Equinox_line/EQNX_2016', '._MLCE20161107_PI_OME.pdf', [b'\x00', b'\x05', b'\x16', b'\x07', b'\x00', b'\x02', b'\x00', b'\x00', b'M', b'a', b'c', b' ', b'O', b'S', b' ', b'X', b' ', b' ', b' ', b''], 'pdf-freetext1']
```
When the translate function is called on it, then the `translate()` function from HeadBytes() 
is called to create a numeric representation of the aforementioned raw data. See: 
```
[ 0  5 22  7  0  2  0  0 77 97 99 32 79 83 32 88 32 32 32  0] 1
```
(where the trailing `1` is the numeric label for `pdf-freetext1`). 

We store the mapping of string labels to numeric labels in the CLASS_TABLE....json
documents stored in `stored_models/class_tables`. 

Finally, the model is trained (in the `train()` function) and then stored
in the `stored_models/trained_classifiers()` directory. 