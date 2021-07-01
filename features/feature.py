import numpy as np

"""
abstract base class for building features from files
Each new set of features should subclass this
"""


class FeatureMaker:
    """
    Takes a open file, and produces features
    """
    def __init__(self):
        self._name = "test"
        self.nfeatures = 1
 
    def get_feature(self, open_file):
        return open_file.name
   
    def translate(self, data_row):
        """
        translate feature from get_feature to 
        a numpy x and y
        """
        return np.zeros(self.nfeatures), 0.0
    
    def get_number_of_features(self):
        return self.nfeatures