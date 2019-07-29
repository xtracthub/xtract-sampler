import numpy as np
import json
import os
from headbytes import HeadBytes
from extpredict import FileReader, SystemReader
from randbytes import RandBytes
from randhead import RandHead
from sklearn import preprocessing


def predict_single_file(filename, trained_classifier, feature):
    """Predicts the type of file.

    filename (str): Name of file to predict the type of.
    trained_classifier: (sklearn model): Trained model.
    feature (str): Type of feature that trained_classifier was trained on.
    """

    with open('new_CLASS_TABLE.json', 'r') as f:
        label_map = json.load(f)
        f.close()
    if feature == "head":
        features = HeadBytes()
    elif feature == "randhead":
        features = RandHead()
    elif feature == "rand":
        features = RandBytes()
    else:
        raise Exception("Not a valid feature set. ")

    reader = FileReader(feature_maker=features, filename=filename)
    reader.run()

    data = [line for line in reader.data][2]
    le = preprocessing.LabelEncoder()  # TODO: Check efficacy. Don't use encoder when training...

    x = np.array(data)
    x = le.fit_transform(x)
    x = [x]

    prediction = trained_classifier.predict(x)

    label = (list(label_map.keys())[list(label_map.values()).index(int(prediction[0]))])
    return label


def predict_directory(dir_name, trained_classifier, feature):
    file_predictions = {}

    with open('new_CLASS_TABLE.json', 'r') as f:
        label_map = json.load(f)
        f.close()
    if feature == "head":
        features = HeadBytes()
    elif feature == "randhead":
        features = RandHead()
    elif feature == "rand":
        features = RandBytes()
    else:
        raise Exception("Not a valid feature set. ")

    reader = SystemReader(feature_maker=features, top_dir=dir_name)
    reader.run()

    for file_data in reader.data:
        data = [line for line in file_data][2]
        le = preprocessing.LabelEncoder()  # TODO: Check efficacy. Don't use encoder when training...

        x = np.array(data)
        x = le.fit_transform(x)
        x = [x]

        prediction = trained_classifier.predict(x)
        print(prediction)
        label = (list(label_map.keys())[list(label_map.values()).index(int(prediction[0]))])
        print(label, os.path.join(file_data[0], file_data[1]))
        file_predictions[os.path.join(file_data[0], file_data[1])] = label

    return file_predictions
