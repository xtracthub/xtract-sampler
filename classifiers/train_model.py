import numpy as np
import json
from random import shuffle
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  plot_confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay # uncomment the corresponding snippets to use


class ModelTrainer(object):
    def __init__(self, reader, C, kernel, iter, degree, penalty, solver, 
                n_estimators, criterion, max_depth, min_sample_split,
                class_table_path, classifier="svc", split=0.8):
        """Initializes the ModelTrainer class.
        reader (list): List of file paths, features, and labels read from a
        label file.
        classifier (str): Type of classifier to use ("svc": support vector
        classifier, "logit": logistic regression, or "rf": random forest).
        split (float): Float between 0 and 1 which indicates how much data to
        use for training. The rest is used as a testing set.
        """
        self.classifier_type = classifier
        self.model = None
        self.class_table = reader.feature.class_table
        self.split = split
        self.params = self.parse_parameters(C, kernel, iter, degree, 
                                                penalty, solver, n_estimators,
                                                criterion, max_depth, min_sample_split)
        

        data = [line for line in reader.data]

        # Puts the data in a different order.
        shuffle(data)

        # Split the data into train and test sets (where split% of the data are for train)
        split_index = int(split * len(data))
        train_data = data[:split_index]  # split% of data.
        test_data = data[split_index:]  # 100% - split% of data.

        # np.zeros: create empty 2D X numpy array (and 1D Y numpy array) for features.
        self.X_train = np.zeros((len(train_data), int(reader.feature.nfeatures)))
        self.Y_train = np.zeros(len(train_data))

        self.X_test = np.zeros((len(test_data), int(reader.feature.nfeatures)))
        self.Y_test = np.zeros(len(test_data))

        groups = [[train_data, self.X_train, self.Y_train],
                  [test_data, self.X_test, self.Y_test]]

        # Here we merge the features into the empty X_train, ..., Y_test objects created above
        # --> Do this for both the train and the test data.
        for group in groups:
            raw_data, X, Y = group
            for i in range(len(raw_data)):
                x, y = reader.feature.translate(raw_data[i])
                X[i] = x
                Y[i] = y

        # model_name = "{}-{}-{}.pkl".format(classifier, feature, timestamp)
        with open(class_table_path, 'w') as class_table:
            json.dump(reader.feature.class_table, class_table)

    def train(self):
        """Trains the model."""
        # TODO: as we fiddle with these, should add options to adjust classifier parameters
        if self.classifier_type == "svc":
            self.model = SVC(kernel=self.params['kernel'], C=self.params['C'], max_iter=self.params['iter'], degree=self.params['degree'])
        elif self.classifier_type == "logit":
            self.model = LogisticRegression(penalty=self.params['penalty'], solver=self.params['solver'], C=self.params['C'], max_iter=self.params['iter'], n_jobs=-1)
        elif self.classifier_type == "rf":
            self.model = RandomForestClassifier(n_estimators=self.params['n_estimators'], criterion=self.params['criterion'], max_depth=self.params['max_depth'], min_samples_split=self.params['min_sample_split'], n_jobs=-1)
        elif self.classifier_type == 'nl-svc':
            self.model =  NuSVC(nu=.01, gamma='auto')
     

        self.model.fit(self.X_train, self.Y_train)

       

    def shuffle(self, split=None):
        """Shuffles the datasets for new trials."""
        if split is None:
            split = self.split

        old_X = np.concatenate((self.X_train, self.X_test), axis=0)
        old_Y = np.concatenate((self.Y_train, self.Y_test), axis=0)

        perm = np.random.permutation(old_Y.shape[0])

        X = old_X[perm]
        Y = old_Y[perm]

        split_index = int(split * X.shape[0])

        self.X_train = X[:split_index]
        self.Y_train = Y[:split_index]

        self.X_test = X[split_index:]
        self.Y_test = Y[split_index:]

    def get_parameters(self):
        return self.params
    
    def parse_parameters(self, C, kernel, iter, degree, penalty, solver,
                        n_estimators, criterion, max_depth, min_sample_split):

        params = dict()

        if self.classifier_type == 'svc':
            params['C'] = C
            params['kernel'] = kernel
            params['iter'] = iter
            params['degree'] = degree

        elif self.classifier_type == 'logit':
            params['C'] = C
            params['iter'] = iter
            params['penalty'] = penalty
            params['solver'] = solver
        
        elif self.classifier_type == 'rf':
            params['n_estimators'] = n_estimators
            params['criterion'] = criterion
            params['max_depth'] = max_depth
            params['min_sample_split'] = min_sample_split
    
        else:
            print("Illegal Classifier.")

        return params

 
