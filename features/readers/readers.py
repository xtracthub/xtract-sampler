import os
import csv
import multiprocessing as mp


"""
This file contains a number of classes that read one file, multiple files, or a LIST of files (available locally), 
and has functions to create (and access) the following: 

--> a list of file directories, filepaths, features, and file labels
"""


class FileReader(object):
    """Takes a single file and turns it into features."""

    def __init__(self, filename, feature_maker):
        """Initializes FileReader class.

        Parameters:
        filename (str): Name of file to turn into features.
        feature_maker (class): An instance of the HeadBytes,
        RandBytes, RandHead class.
        """
        if not os.path.isfile(filename):
            raise FileNotFoundError("%s is not a valid file" % filename)

        self.filename = filename
        self.feature = feature_maker
        self.data = []

    def handle_file(self, filename):
        """Extract features from a file.

        Parameter:
        filename (str): Name of file to extract features from.

        Return:
        (list): List of features and file extension of filename.
        """
        try:
            with open(filename, "rb") as open_file:
                extension = get_extension(filename)
                features = self.feature.get_feature(open_file)
                #print("basename: ", os.path.basename(filename))
                return [os.path.basename(filename), filename, features, extension]

        except (FileNotFoundError, PermissionError):
            pass

    def run(self):
        self.data = self.handle_file(self.filename)


class NaiveTruthReader(object):
    """Takes a .csv file of filepaths and file labels and returns a
    list of file directories, filepaths, features, and file labels.
    """
    def __init__(self, feature_maker, labelfile="naivetruth.csv"):
        """Initializes NaiveTruthReader class.

        Parameters:
        feature_maker (str): An instance of a FileFeature class to extract
        features with (HeadBytes, RandBytes, RandHead, Ngram, RandNgram).
        labelfile (.csv file): .csv file containing filepaths and labels.

        Return:
        (list): List of filepaths, file names, features, and labels.
        """
        self.feature = feature_maker
        self.data = []
        self.labelfile = labelfile

    def extract_row_data(self, row):
        try:
            with open(row["path"], "rb") as open_file:
                features = self.feature.get_feature(open_file)
                row_data = ([os.path.dirname(row["path"]),
                            os.path.basename(row["path"]), features,
                            row["file_label"]])
                return row_data
        except (FileNotFoundError, PermissionError):
            print("Could not open %s" % row["path"])

    def run(self):
        labelf = open(self.labelfile, "r")

        reader = csv.DictReader(labelf)
        pools = mp.Pool(processes=mp.cpu_count())
        self.data = pools.map(self.extract_row_data, reader)
        pools.close()
        pools.join()
        for idx, item in enumerate(self.data):
            self.data[idx] = item
    
    def get_feature_maker(self):
        return self.feature


def get_extension(filename):
    """Retrieves the extension of a file.

    Parameter:
    filename (str): Name of file you want to get extension from.

    Return:
    (str): File extension of filename.
    """
    if "." not in filename:
        return "None"
    return filename[filename.rfind("."):]
