from os.path import getsize
from features.feature import FeatureMaker
from random import randint
from features.headbytes import HeadBytes


class RandHead(FeatureMaker):
    """Retrieves bytes from the head of a file and random bytes from
    subsequent sections."""
    def __init__(self, head_size=512, rand_size=512):
        """Initializes RandHead class.

        Parameters:
        head_size (int): Number of bytes to retrieve from the header of
        the file.
        rand_size (int): Number of random of bytes to retrieve from
        the file.
        """
        self.name = "randhead"
        self.head_size = head_size
        self.rand_size = rand_size
        self.nfeatures = head_size + rand_size
        self.class_table = {}
        self._head = HeadBytes(head_size=self.head_size)

    def get_feature(self, open_file):
        """Retrieves head_size number of bytes from the header of
        open_file and retrieves rand_size number of random bytes from
        open_file.

        Parameter:
        open_file (file): An opened file to retrieve data from.

        Return:
        sample_bytes (list): A list of head_size number of bytes from
        the header of open_file and rand_size number of random bytes
        from open_file.
        """
        head = self._head.get_feature(open_file)
        
        size = getsize(open_file.name)

        if size == 0:
            return [b'' for i in range(self.nfeatures)]

        if size > self.head_size:
            rand_index = [randint(self.head_size, size-1) for _ in range(self.rand_size)]
        else: # possibly the right way??
            rand_index = [randint(0, size-1) for _ in range(self.rand_size)]

        rand_index.sort()
        sample_bytes = head

        for index in rand_index:
            
            open_file.seek(index)
            sample_bytes.append(open_file.read(1))

        return sample_bytes

    def translate(self, entry):
        x, y = self._head.translate(entry)
        self.class_table = self._head.class_table
        return x, y
