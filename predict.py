import numpy as np
import json
from headbytes import HeadBytes
from extpredict import FileReader
from randbytes import RandBytes
from randhead import RandHead
from sklearn import preprocessing


def predict_single_file(filename, trained_classifier, feature):
    """Predicts the type of file.

    filename (str): Name of file to predict the type of.
    trained_classifier: (sklearn model): Trained model.
    feature (str): Type of feature that trained_classifier was trained on.
    """

    with open('CLASS_TABLE.json', 'r') as f:
        label_map = json.load(f)
        f.close()

    if feature == "head":
        features = HeadBytes()
    elif feature == "randhead":
        features = RandHead()
    elif feature == "rand":
        features = RandBytes()
    elif feature == "randngram":
        features = RandNgram()
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
