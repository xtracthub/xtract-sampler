import argparse
import time
import datetime
import pickle as pkl
import json
import os

from headbytes import HeadBytes
from extpredict import NaiveTruthReader
from train_model import ModelTrainer
from test_model import score_model
from randbytes import RandBytes
from randhead import RandHead
from consec_bytes import ConsecBytes
from predict import predict_single_file, predict_directory

# Current time for documentation purposes
current_time = datetime.datetime.today().strftime('%Y-%m-%d')


def main():
    parser = argparse.ArgumentParser(description='Run file classification experiments')

    parser.add_argument("--dirname", type=str, help="directory of files to predict", default=None)
    parser.add_argument("--n", type=int, default=1,
                        help="number of trials", dest="n")
    parser.add_argument("--classifier", type=str,
                        help="model to use: svc, logit, rf")
    parser.add_argument("--feature", type=str, default="head",
                        help="feature to use: head, rand, randhead, consec_bytes")
    parser.add_argument("--split", type=float, default=0.8,
                        help="test/train split ratio", dest="split")
    parser.add_argument("--head_bytes", type=int, default=512,
                        dest="head_bytes",
                        help="size of file head in bytes, default 512")
    parser.add_argument("--rand_bytes", type=int, default=512,
                        dest="rand_bytes",
                        help="number of random bytes, default 512")
    parser.add_argument("--consec_bytes", help="number of total consecutive bytes, default 512",
                        default=512)
    parser.add_argument("--nconsec", help="number of consecutive bytes, default 4",
                        default=4)
    parser.add_argument("--predict_file", type=str, default=None,
                        help="file to predict based on a classifier and a "
                             "feature")
    parser.add_argument("--trained_classifier", type=str,
                        help="trained classifier to predict on",
                        default='rf-head-default.pkl')
    args = parser.parse_args()

    if args.predict_file is not None:
        try:
            with open(args.trained_classifier, 'rb') as classifier_file:
                trained_classifier = pkl.load(classifier_file)
        except:
            print("Invalid trained classifier")

        if args.feature not in ["head", "rand", "randhead", "consec_bytes"]:
            print("Invalid feature option %s" % args.feature)
            return
        print(predict_single_file(args.predict_file, trained_classifier,
                                  args.feature))
    elif args.dirname is not None:
        try:
            with open(args.trained_classifier, 'rb') as classifier_file:
                trained_classifier = pkl.load(classifier_file)
        except:
            print("Invalid trained classifier")

        if args.feature not in ["head", "rand", "randhead", "consec_bytes"]:
            print("Invalid feature option %s" % args.feature)
            return
        with open("prediction_results.json", 'w') as prediction_file:
            json.dump(predict_directory(args.dirname, trained_classifier, args.feature, head_bytes=args.head_bytes,
                                        rand_bytes=args.rand_bytes), prediction_file)
    else:
        if args.classifier not in ["svc", "logit", "rf"]:
            print("Invalid classifier option %s" % args.classifier)
            return

        if args.feature == "head":
            features = HeadBytes(head_size=args.head_bytes)
        elif args.feature == "rand":
            features = RandBytes(number_bytes=args.rand_bytes)
        elif args.feature == "randhead":
            features = RandHead(head_size=args.head_bytes,
                                rand_size=args.rand_bytes)
        elif args.feature == "consec_bytes":
            features = ConsecBytes(number_bytes=args.consec_bytes,
                                   n_consec=args.nconsec)
        else:
            print("Invalid feature option %s" % args.feature)
            return

        reader = NaiveTruthReader(features)
        experiment(reader, args.classifier, args.feature, args.n,
                   split=args.split)


def experiment(reader, classifier_name, features, trials, split):
    """Trains classifier_name on features from files in reader trials number
    of times and saves the model and returns training and testing data.

    Parameters:
    reader (list): List of file paths, features, and labels read from a
    label file.
    classifier_name (str): Type of classifier to use ("svc": support vector
    classifier, "logit": logistic regression, or "rf": random forest).
    features (str): Type of features to train on (head, rand, randhead,
    ngram, randngram).
    outfile (str): Name of file to write outputs to.
    trials (int): Number of times to train a model with randomized features
    for each training.
    split (float): Float between 0 and 1 which indicates how much data to
    use for training. The rest is used as a testing set.

    Return:
    (pkl): Writes a pkl file containing the model.
    (json): Writes a json named outfile with training and testing data.
    """
    read_start_time = time.time()
    print("reading")
    reader.run()
    print("done reading")
    read_time = time.time() - read_start_time
    print(read_time)
    classifier = ModelTrainer(reader, classifier=classifier_name, split=split)

    for i in range(trials):
        print("Starting trial {} out of {} for {} {}".format(i, trials,
                                                             classifier_name,
                                                             features))
        classifier_start = time.time()
        print("training")
        classifier.train()
        print("done training")
        accuracy = score_model(classifier.model, classifier.X_test,
                               classifier.Y_test)
        classifier_time = time.time() - classifier_start

        model_name = "{}-{}-trial{}-{}.pkl".format(classifier_name, features,
                                                   i + 1, current_time)
        outfile_name = "{}-{}-{}.json".format(classifier_name, features,
                                              current_time)

        with open(model_name, "wb") as model_file:
            pkl.dump(classifier.model, model_file)
        with open(outfile_name, "a") as data_file:
            output_data = {"Classifier": classifier_name,
                           "Feature": features,
                           "Trial": i,
                           "Read time": read_time,
                           "Train and test time": classifier_time,
                           "Model accuracy": accuracy,
                           "Model size": os.path.getsize(model_name)}
            json.dump(output_data, data_file)

        if i != trials-1:
            classifier.shuffle()


if __name__ == '__main__':
    main()
