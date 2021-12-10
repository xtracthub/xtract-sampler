
import numpy as np
import pandas as pd
import pickle as pkl

# Import Sk-Learn Utilities
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

# Import additional ML Utilities
from xgboost.sklearn import XGBRegressor
from scipy.stats import pearsonr
from canova_source import canova


def generate_regressors(data_file, extractor, k=2):
    """
    Function to generate best models for predicting given metrics.
    Depending on context, uses canova and R^2 values to determine 'best'.

    :param data_file --> (str) path to a CSV file with [file_path,extraction_time,extraction_size,file_size]
    :param extractor --> (str) name of the extractor for which we are training the model
    :param k --> (int) positive integer for k-fold cross-validation.

    :returns model_file pickled dictionary of metric : Pipeline[sk-learn] objects.
    """

    data_df = pd.read_csv(data_file)

    metrics = list(data_df.columns.values)[1:3]
    model_pile = dict.fromkeys(metrics, None)  # assume filename is first col

    print(f"List of metrics in model: {metrics}")

    for metric in metrics:
        linear_pipelines = dict()
        nonlinear_pipelines = dict()

        # Assemble linear pipelines with sk-learn's StandardScaler
        # Pipeline info:
        #   https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
        # StandardScaler info:
        #   https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
        linear_pipelines['ScaledLR'] = Pipeline([('Scaler', StandardScaler()), ('LR', LinearRegression())])
        linear_pipelines['ScaledLASSO'] = Pipeline([('Scaler', StandardScaler()), ('LASSO', Lasso())])
        linear_pipelines['ScaledSVR'] = Pipeline([('Scaler', StandardScaler()), ('CART', SVR())])

        # Now also create pipelines for non-linear models (also with standard scaler).
        nonlinear_pipelines['ScaledXGB'] = \
            Pipeline([('Scaler', StandardScaler()), ('EN', XGBRegressor())])
        nonlinear_pipelines['ScaledKR'] = \
            Pipeline([('Scaler', StandardScaler()), ('KNN', KernelRidge())])
        nonlinear_pipelines['ScaledGBM'] = \
            Pipeline([('Scaler', StandardScaler()), ('GBM', GradientBoostingRegressor())])

        # X is file_size -- trying to use file size to predict...
        # Y is the metrics. Such sa extraction_time and extraction_size
        X = data_df['file_size'].to_numpy()
        Y = data_df[metric].to_numpy()

        # show the linear correlation between X and Y (just 1 Y metric)
        #   https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
        r, _ = pearsonr(X, Y)  # disregard p-value
        #
        canova_value = canova(X, Y)

        X = X.reshape(-1, 1)

        linear_scores = dict()
        nonlinear_scores = dict()

        if r >= 0.4:
            # correlated
            for name, model in linear_pipelines.items():
                # Split into ten folds and hope the bias isn't too high (there aren't TOO many files)
                k_fold = KFold(n_splits=k, shuffle=True)
                r2_score = np.mean(cross_val_score(model, X, Y, cv=k_fold, scoring='r2'))

                print(f"{name} -- r2_score: {r2_score}")
                if r2_score > 0:
                    linear_scores[name] = r2_score
                    print("r2_score: ", r2_score)

        if canova_value >= 0.4:
            for name, model in nonlinear_pipelines.items():
                # Split into ten folds and hope the bias isn't too high (there aren't TOO many files)
                k_fold = KFold(n_splits=k, shuffle=True)
                r2_score = np.mean(cross_val_score(model, X, Y, cv=k_fold, scoring='r2'))

                if r2_score > 0:
                    nonlinear_scores[name] = r2_score
                    print("r2_score: ", r2_score)

        if len(linear_scores) == 0 and len(nonlinear_scores) == 0:  # in cases all regressions are bad
            print(f"[generate regressors] ALL REGRESSIONS ARE BAD.")
            model_pile[metric] = data_df[metric].mean()
        else:
            model_pile[metric] = pick_best_model(
                linear_scores,
                nonlinear_scores,
                X,
                Y,
                linear_pipelines,
                nonlinear_pipelines)

    pkl.dump(model_pile, open("Model_Pile_" + extractor + ".pkl", "wb"))
    return model_pile


def pick_best_model(linear_scores, nonlinear_scores, X, Y, linear_pipelines, nonlinear_pipelines):
    """
    Function to input linear_scores (R^2), nonlinear_scores (canova), data, and model pipelines
    Outputs the best model.
    """

    max_score = 0

    model = None
    is_linear_model = None

    # Iterate over all linear models -- if a score is > 0, then is_linear_model is True
    for name, score in linear_scores.items():
        if score > max_score:
            max_score = score
            model = name
            is_linear_model = True
    # Iterate over all non-linear models. If Better than best linear models, then is_linear_model is False
    for name, score in nonlinear_scores.items():
        if score > max_score:
            max_score = score
            model = name
            is_linear_model = False

    assert model is not None
    assert is_linear_model is not None

    print("Model: ", model)
    print("Score: ", max_score)
    # TODO: Can we directly compare canova with R^2?
    if is_linear_model:
        best_model = linear_pipelines[model]
    else:
        best_model = nonlinear_pipelines[model]
    best_model.fit(X, Y)
    return best_model


if __name__ == "__main__":
    model_pkl = generate_regressors("test_data_file_linear_fit.csv", "keyword")
    print(f"Model PKL: {model_pkl}")
