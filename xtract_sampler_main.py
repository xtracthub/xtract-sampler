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
from predict import predict_single_file, predict_directory
# from automated_training import write_naive_truth
# from cloud_automated_training import write_naive_truth

# Current time for documentation purposes
current_time = datetime.datetime.today().strftime('%Y-%m-%d')


def experiment(reader, classifier_name, features, trials, split, model_name, features_outfile):
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
    features_oufile (str): File path to .pkl file where a reader class is
    stored.

    Return:
    (pkl): Writes a pkl file containing the model.
    (json): Writes a json named outfile with training and testing data.
    """
    read_start_time = time.time()
    if features_outfile is None:
        reader.run()
    read_time = time.time() - read_start_time
    classifier = ModelTrainer(reader, classifier=classifier_name, split=split)

    # print(classifier)
    # exit()

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

        outfile_name = "{}-info.json".format(os.path.splitext(model_name)[0])

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


def extract_sampler(mode='train', classifier='rf', feature='head', model_name=None, n=1, head_bytes=512, rand_bytes=512,
                    split=0.8, label_csv=None, dirname=None, predict_file=None,
                    trained_classifier='rf-head-default.pkl', results_file='sampler_results.json',
                    csv_outfile='naivetruth.csv', features_outfile=None):
    if mode == 'predict' and predict_file is not None:
        try:
            with open(trained_classifier, 'rb') as classifier_file:
                trained_classifier = pkl.load(classifier_file)
        except:
            print("Invalid trained classifier")

        print(trained_classifier)
        exit()

        if feature not in ["head", "rand", "randhead"]:
            print("Invalid feature option %s" % feature)
            return
        # with open(results_file, 'w') as prediction_file:
        #     json.dump(predict_single_file(dirname, trained_classifier, feature, head_bytes=head_bytes,
        #                                   rand_bytes=rand_bytes), prediction_file)
        t0 = time.time()
        prediction = predict_single_file(predict_file, trained_classifier,
                                         feature)
        meta = {"sampler": {predict_file: prediction}, "extract time": time.time() - t0}
        print(meta)
        return meta

    elif mode == 'predict' and dirname is not None:
        try:
            with open(trained_classifier, 'rb') as classifier_file:
                trained_classifier = pkl.load(classifier_file)
        except:
            print("Invalid trained classifier")

        if feature not in ["head", "rand", "randhead"]:
            print("Invalid feature option %s" % feature)
            return
        # with open(results_file, 'w') as prediction_file:
        #     json.dump(predict_directory(dirname, trained_classifier, feature, head_bytes=head_bytes,
        #                                 rand_bytes=rand_bytes), prediction_file)
        t0 = time.time()
        predictions = predict_directory(dirname, trained_classifier, feature, head_bytes=head_bytes,
                                        rand_bytes=rand_bytes)
        meta = {"sampler": predictions, "extract time": time.time() - t0}
        print(meta)
        return meta

    elif mode == 'train':
        if classifier not in ["svc", "logit", "rf"]:
            print("Invalid classifier option %s" % classifier)
            return

        if feature == "head":
            features = HeadBytes(head_size=head_bytes)
        elif feature == "rand":
            features = RandBytes(number_bytes=rand_bytes)
        elif feature == "randhead":
            features = RandHead(head_size=head_bytes,
                                rand_size=rand_bytes)
        else:
            print("Invalid feature option %s" % feature)
            return

        if model_name is None:
            model_name = "{}-{}-{}.pkl".format(classifier, feature, current_time)

        if os.path.exists(features_outfile):
            try:
                with open(features_outfile, 'rb') as f:
                    reader = pkl.load(f)
            except:
                print("Invalid features outfile")
                return
        else:
            reader = NaiveTruthReader(features, labelfile=label_csv)

        experiment(reader, classifier, feature, n,
                   split, model_name, features_outfile)

    elif mode == 'labels_features':

        # write_naive_truth(csv_outfile, dirname, multiprocess=True, chunksize=1, n=1000)

        if feature == "head":
            features = HeadBytes(head_size=head_bytes)
        elif feature == "rand":
            features = RandBytes(number_bytes=rand_bytes)
        elif feature == "randhead":
            features = RandHead(head_size=head_bytes,
                                rand_size=rand_bytes)
        else:
            print("Invalid feature option %s" % feature)
            return
        print("Running naive truth reader")
        reader = NaiveTruthReader(features,  labelfile=csv_outfile)
        reader.run()
        
        print("Saving features to: {}".format(features_outfile))

        try:
            if os.path.getsize(features_outfile) > 0:
                with open(features_outfile, 'rb') as f:
                    reader_object = pkl.load(f)
                with open(features_outfile, 'wb') as f:
                    reader_object.data.extend(reader.data)
                    pkl.dump(reader_object, f)
            else:
                with open(features_outfile, 'ab') as f:
                    pkl.dump(reader, f)
        except:
            print("There was an exception!")
            with open(features_outfile, 'ab') as f:
                pkl.dump(reader, f)

        print("Done saving features")

    else:
        print("Invalid mode")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run file classification experiments')

    parser.add_argument("--dirname", type=str, help="directory of files to predict if mode is predict"
                                                    "or directory to get labels and features of", default=None)
    parser.add_argument("--n", type=int, default=1,
                        help="number of trials", dest="n")
    parser.add_argument("--classifier", type=str,
                        help="model to use: svc, logit, rf")
    parser.add_argument("--feature", type=str, default="head",
                        help="feature to use: head, rand, randhead")
    parser.add_argument("--split", type=float, default=0.8,
                        help="test/train split ratio", dest="split")
    parser.add_argument("--head_bytes", type=int, default=512,
                        dest="head_bytes",
                        help="size of file head in bytes, default 512")
    parser.add_argument("--rand_bytes", type=int, default=512,
                        dest="rand_bytes",
                        help="number of random bytes, default 512")
    parser.add_argument("--predict_file", type=str, default=None,
                        help="file to predict based on a classifier and a "
                             "feature")
    parser.add_argument("--trained_classifier", type=str,
                        help="trained classifier to predict on",
                        default='rf-head-default.pkl')
    parser.add_argument("--results_file", type=str,
                        default="sampler_results.json", help="Name for results file if predicting")
    parser.add_argument("--label_csv", type=str, help="Name of csv file with labels",
                        default="automated_training_results/naivetruth.csv")
    parser.add_argument("--csv_outfile", type=str, help="file to write labels to",
                        default='naivetruth.csv')
    parser.add_argument("--model_name", type=str, help="Name of model",
                        default=None)
    parser.add_argument("--mode", type=str, help="Mode (train, predict, labels_features)",
                        default='train')
    parser.add_argument("--features_outfile", type=str, help="file to write features to if mode is labels_feautres"
                                                             "else it's a pkl with a reader object",
                        default=None)
    args = parser.parse_args()

    meta = extract_sampler(args.mode, args.classifier, args.feature, args.model_name, args.n, args.head_bytes,
                           args.rand_bytes, args.split, args.label_csv, args.dirname, args.predict_file,
                           args.trained_classifier, args.results_file, args.csv_outfile, args.features_outfile)
    print(meta)

