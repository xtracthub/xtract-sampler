import os
import csv
import numpy as np


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
                return (['/home/skluzacek', 'newfile.csv', features, extension])

        except (FileNotFoundError, PermissionError):
            pass

    def run(self):
        self.data = self.handle_file(self.filename)


class SystemReader(object):
    """Traverses file system, and produces initial dataset for prediction."""

    def __init__(self, top_dir, feature_maker):
        """Initializes SystemReader class.

        top_dir (str): The starting directory of files to get
        features from.
        feature_maker (class): An instance of the HeadBytes,
        RandBytes, RandHead class.
        """
        if not os.path.isdir(top_dir):
            raise NotADirectoryError("%s is not a valid directory" % top_dir)

        self.dirname = top_dir
        self.feature = feature_maker
        self.data = []
        self.next_dirs = []

    def handle_file(self, filename, current_dir):
        """Appends current_dir, filename, features of filename and
        extension of filename to self.data.

        Parameters:
        filename (str): Name of file to extract features from.
        current_dir (str): Name of current directory that filename is
        in.
        """
        # at some point we may want to parallelize fs traversal, to do that
        # we could make this a standalone function and use pool.map

        try:
            with open(os.path.join(current_dir, filename), "rb") as open_file:

                extension = get_extension(filename)
                features = self.feature.get_feature(open_file)
                self.data.append([current_dir, filename, features, extension])

        except (FileNotFoundError, PermissionError):
            pass

    def parse_dir(self, dirname):
        """Parse a directory with path dirname, add subdirectories to
        the list to be processed, and extract features from files.

        Parameter:
        dirname (str): Name of directory to parse.
        """
        # at some point we may want to parallelize fs traversal, to do that
        # we could make this a standalone function and use pool.map
        files = []

        for name in os.listdir(dirname):
            if name[0] == ".":
                continue # exclude hidden files and dirs for time being
            if os.path.isfile(os.path.join(dirname, name)):
                files.append(name)
            elif os.path.isdir(os.path.join(dirname, name)):
                self.next_dirs.append(os.path.join(dirname, name))

        for filename in files:
            self.handle_file(filename, dirname)

    def run(self):
        """Extract features from all files in top_dir."""
        self.next_dirs = [self.dirname]

        while self.next_dirs:
            dirname = self.next_dirs.pop(0)
            self.parse_dir(dirname)


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
                print(np.array(row_data).shape)
                return row_data
        except (FileNotFoundError, PermissionError):
            print("Could not open %s" % row["path"])

    def run(self):
        labelf = open(self.labelfile, "r")

        reader = csv.DictReader(labelf)
        # pools = mp.Pool()
        #
        # self.data = pools.map(self.extract_row_data, reader)
        # pools.close()
        # pools.join()
        # for idx, item in enumerate(self.data):
        #     self.data[idx] = item
        # print(self.data[0])
        # print(np.array(self.data).shape)

        for idx, row in enumerate(reader):
            try:
                with open(row["path"], "rb") as open_file:
                    features = self.feature.get_feature(open_file)
                    row_data = ([os.path.dirname(row["path"]),
                                 os.path.basename(row["path"]), features,
                                 row["file_label"]])
                    print(idx)
                    self.data.append(row_data)
            except (FileNotFoundError, PermissionError):
                print("Could not open %s" % row["path"])

        print(np.array(self.data).shape)


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

