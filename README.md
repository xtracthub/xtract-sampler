# xtract-sampler
ML code to sample a file based on cheap, easily-attainable features of a file. 

To see documentation about the code itself (re: features, readers, etc., please see the README file in classifiers/README.md)

## Training a model using a .csv:
`python xtract_sampler_main.py --mode train --classifier ex1 --feature ex2 --label_csv ex3`
- `ex1` should be either rf, svc, or logit for a random forest, support vector classification, or logistic regression model.
- `ex2` should be either head, rand, randhead to set the features as bytes from the head of the file, random bytes, or a mixture of both.
- `ex3` is the path to a .csv file with the file path, file size, and file label for files to train on.
- Additional `--head_bytes` and `--rand_bytes` parameters can be passed to specify the number of bytes to take from the file (the default is 512 bytes if these parameters aren't passed).

## Predicting using a trained classifier:
`python xtract_sampler_main.py --mode predict --trained_classifier ex1 --feature ex2 --predict_file ex3`
- `ex1` is the path to a trained classifier, trained using the training mode of xtract_sampler_main.py.
- `ex2` is the type of feature that `ex1` was trained on (head, rand, randhead).
    - **Note**: If a `--head_bytes` or `--rand_bytes` value was passed during training, the same value should be passed during                  predicting. 
- `ex3` is the path to the file to predict on.
    - Alternatively, to predict on a directory, use `--dirname ex3` instead of `--predict_file ex3`.

## Running two-phase automated training (NEEDS TO BE TESTED):
Two-phase automated training allows users to generate labels and save features for multiple directories before training on those features and labels.
1. `python xtract_sampler_main.py --mode labels_features --dirname ex1 --features_outfile ex2 --csv_outfile ex3 --features ex4`
    - `ex1` is the directory to generate labels from and to grab features from.
    - `ex2` is the name/path to the .pkl file to write file features to. 
    - `ex3` is the name/path to the .csv file to write labels to.
    - `ex4` should be either head, rand, randhead to set the features as bytes from the head of the file, random bytes, or a mixture of both.
    - Additional `--head_bytes` and `--rand_bytes` parameters can be passed to specify the number of bytes to take from the file (the default is 512 bytes if these parameters aren't passed).
2. Repeat step 1 with as many directories as you want. However, `--features_outfile` and `--features` must always be the same. Additionally if `--head_bytes` or `--rand_bytes` is passed, they must stay the same too.
3. `python xtract_sampler_main.py --mode train --classifier ex1 --features ex2 --features_outfile ex3`
    - `ex1` should be either rf, svc, or logit for a random forest, support vector classification, or logistic regression model.
    - `ex2` should be either head, rand, randhead for the features to be bytes from the head of the file, random bytes, or a mixture of both.
    - `ex3` is the name/path of the .pkl file passed to `--features_outfile` in steps 1 and 2.
        - **Note**: If a `--head_bytes` or `--rand_bytes` value was passed during steps 1 and 2, the same value should be passed here.

## Where are files saved?
- Models created using the training mode will be saved under the name `stored_models/trained_classifiers/<classifier>-<feature>-<date>.pkl` where the classifier and feature are the values passed to the command line and date is the current date. Training a model will also create a .json file named `stored_models/trained_classifiers/<classifier>-<feature>-<date>.json` that will contain training times and accuracy results about the trained model. **CURRENTLY UNSUPPORTED**: To change the model name, pass `--model_name ex1` where ex1 is the name of the file to save the model.
- **Currently Unsupported**: Predictions from the prediction mode will be saved under the name `sampler_results.json`. To change this, pass `--results_file ex1` where ex1 is the name of the file to save prediction results. 
