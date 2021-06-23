import numpy as np
from features.feature import FeatureMaker


class HeadBytes(FeatureMaker):
    """Retrieves bytes from the head of a file."""
    def __init__(self, head_size=512):
        """Initializes HeadBytes class.

        Parameters:
        head_size (int): Number of bytes to get from header.
        """
        self.name = "head"
        self.head_size = head_size
        self.nfeatures = head_size
        self.class_table = {}
 
    def get_feature(self, open_file):
        """Retrieves the first head_size number of bytes from a file.

        Parameter:
        open_file (file): An opened file to retrieve data from.

        Return:
        head (list): A list of the first head_size bytes in
        open_file.
        If there are less than head_size bytes in
        open_file, the remainder of head is filled with empty bytes.
        """
        byte = open_file.read(1) 
        read = 1  
        head = [] 

        while byte and read < self.head_size:

            head.append(byte)
            read += 1
            byte = open_file.read(1)

        if len(head) < self.head_size:
            head.extend([b'' for i in range(self.head_size - len(head))])
        assert len(head) == self.head_size
        return head

    def translate(self, entry):
        """Translates a feature into an integer.

        Parameter:
        entry (byte): A feature.

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

