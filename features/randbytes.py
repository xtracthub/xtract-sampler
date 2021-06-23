from os.path import getsize
from features.feature import FeatureMaker
from random import randint
import numpy as np


class RandBytes(FeatureMaker):
    """Retrieves random bytes from a file."""
    def __init__(self, number_bytes=512):
        """Initializes RandBytes class.

        Parameters:
        number_bytes (int): Number of random bytes to get.
        """
        self.name = "rand"
        self.nfeatures = number_bytes

        self.class_table = {}

    def get_feature(self, open_file):
        """Retrieves number_bytes number of random bytes from open_file.

        Parameter:
        open_file (file): An opened file to retrieve data from.

        Return:
        sample_bytes (list): A list of number_bytes number of random
        bytes from open_file.
        """
        size = getsize(open_file.name)

        if size == 0:
            return [b'' for i in range(self.nfeatures)]
        else:
            rand_index = [randint(0, size-1) for _ in range(self.nfeatures)]

        # For files where size < nfeatures, this will oversample.
        # This may be something to look out for though. 
      
        rand_index.sort()
        sample_bytes = []

        for index in rand_index:

            open_file.seek(index)
            sample_bytes.append(open_file.read(1))
        return sample_bytes

    def translate(self, entry):
        """Translates a feature into an integer.

        Parameter:
        entry (list): A list of a file path, file name, list of bytes, and a label.

        Return:
        (tuple): 2-tuple of a numpy array containing an integer version of
        entry and a dictionary of labels and indices.
        """
        x = [int.from_bytes(c, byteorder="big") for c in entry[2]]

        try:
            y = self.class_table[entry[-1]]
        except KeyError:
            self.class_table[entry[-1]] = len(self.class_table) + 1
            y = self.class_table[entry[-1]]

        return np.array(x), y
