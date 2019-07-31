from feature import FeatureMaker
import os
import numpy as np
import math


class ConsecBytes(FeatureMaker):
    """Retrieves consecutive bytes from a file"""
    def __init__(self, number_bytes=512, n_consec=4):
        """Initializes the ConsecBytes class.

        Parameter:
        number_bytes (int): Number of total bytes to retrieve from a file.
        n_consec (int): Number of consecutive bytes to retrieve.
        """
        self.name = "consec"
        self.nfeatures = number_bytes
        self.nconsec = n_consec
        self.class_table = {}

    def get_feature(self, open_file):
        """Retrieves number_bytes number of random bytes from open_file.

        Parameter:
        open_file (file): An opened file to retrieve data from.

        Return:
        sample_bytes (list): A list of number_bytes number of random
        bytes from open_file.
        """
        sample_bytes = []
        file_bytes = [open_file.read(1) for i in range(os.path.getsize(open_file.name))]

        if len(file_bytes) == 0:
            raise FileNotFoundError
        elif len(file_bytes) >= self.nfeatures:
            rand_index = np.random.choice(range(len(file_bytes)), math.floor(self.nfeatures / self.nconsec))
            rand_index.sort()

            for rand_num in rand_index:
                if rand_num > len(file_bytes) - self.nconsec:
                    small_bytes_list = file_bytes[rand_num:]
                    small_bytes_list.extend([b'' for i in range(self.nconsec - (len(file_bytes) - rand_num))])
                    sample_bytes.append(small_bytes_list)
                else:
                    sample_bytes.append(file_bytes[rand_num: rand_num + self.nconsec])
        else:
            for i in range(0, len(file_bytes), self.nconsec):
                sample_bytes.append(file_bytes[i: i + 4])
                if len(sample_bytes[-1]) < self.nconsec:
                    sample_bytes[-1].extend([b'' for i in range(self.nconsec - len(sample_bytes[-1]))])
            for i in range(math.floor(self.nfeatures / self.nconsec) - len(sample_bytes)):
                sample_bytes.append([b'' for i in range(self.nconsec)])

        return sample_bytes

    def translate(self, entry):
        """Translates a feature into an integer.

        Parameter:
        entry (list): A list of a file path, file name, list of bytes, and a label.

        Return:
        (tuple): 2-tuple of a numpy array containing an integer version of
        entry and a dictionary of labels and indices.
        """
        x = []

        for byte_group in entry[2]:
            x.append([int.from_bytes(c, byteorder="big") for c in byte_group])

        try:
            y = self.class_table[entry[-1]]
        except KeyError:
            self.class_table[entry[-1]] = len(self.class_table) + 1
            y = self.class_table[entry[-1]]

        return np.array(x), y

