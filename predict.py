import numpy as np
import json
import os
from headbytes import HeadBytes
from extpredict import FileReader, SystemReader
from randbytes import RandBytes
from randhead import RandHead


def predict_single_file(filename, trained_classifier, feature, head_bytes=512, rand_bytes=512):
    """Predicts the type of file.

    filename (str): Name of file to predict the type of.
    trained_classifier: (sklearn model): Trained model.
    feature (str): Type of feature that trained_classifier was trained on.
    """

    with open('CLASS_TABLE.json', 'r') as f:
        label_map = json.load(f)
        f.close()
    if feature == "head":
        features = HeadBytes(head_size=head_bytes)
    elif feature == "randhead":
        features = RandHead(head_size=head_bytes, rand_size=rand_bytes)
    elif feature == "rand":
        features = RandBytes(number_bytes=rand_bytes)
    else:
        raise Exception("Not a valid feature set. ")

    reader = FileReader(feature_maker=features, filename=filename)
    reader.run()

    data = [line for line in reader.data][2]
    x = np.array([int.from_bytes(c, byteorder="big") for c in data])
    x = [x]

    prediction = trained_classifier.predict(x)

    label = (list(label_map.keys())[list(label_map.values()).index(int(prediction[0]))])
    return label


def predict_directory(dir_name, trained_classifier, feature, head_bytes=512, rand_bytes=512):
    file_predictions = {}

    with open('new_CLASS_TABLE.json', 'r') as f:
        label_map = json.load(f)
        f.close()
    if feature == "head":
        features = HeadBytes(head_size=head_bytes)
    elif feature == "randhead":
        features = RandHead(head_size=head_bytes,
                            rand_size=rand_bytes)
    elif feature == "rand":
        features = RandBytes(number_bytes=rand_bytes)
    else:
        raise Exception("Not a valid feature set. ")

    reader = SystemReader(feature_maker=features, top_dir=dir_name)
    reader.run()

    for file_data in reader.data:

        data = [line for line in file_data][2]

        x = np.array([int.from_bytes(c, byteorder="big") for c in data])
        x = [x]

        prediction = trained_classifier.predict(x)
        label = (list(label_map.keys())[list(label_map.values()).index(int(prediction[0]))])
        file_predictions[os.path.join(file_data[0], file_data[1])] = label

    return file_predictions
