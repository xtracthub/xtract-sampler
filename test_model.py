
from sklearn.metrics import precision_score

def score_model(model, X_test, y_test):
    """Scores the model.

    Parameters:
    model (sklearn model): Trained sklearn model (SVC, LinearRegression,
    RandForestClassifier).
    X_test (list): List of files to test on.
    y_test (list): List of labels for X_test.

    Return:
    (float): The percentage of files from X_test that model was able to
    correctly classify.
    """
    # avg_prec_score = precision_score(X_test, y_test)
    # print(f"Model precision: {avg_prec_score}")

    return model.score(X_test, y_test)

