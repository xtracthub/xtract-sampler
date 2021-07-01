
from sklearn.metrics import precision_score, recall_score

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


    y_pred = model.predict(X_test)
    avg_prec_score_micro = precision_score(y_pred, y_test, average='micro')
    avg_prec_score_macro = precision_score(y_pred, y_test, average='macro')
    avg_prec_score_weighted = precision_score(y_pred, y_test, average='weighted')
    #avg_prec_score_samples = precision_score(y_pred, y_test, average='samples')
    avg_prec_score_overall = (avg_prec_score_micro + avg_prec_score_macro + avg_prec_score_weighted) / 3

    print(f"Model precision (micro): {avg_prec_score_micro}")
    print(f"Model precision (macro): {avg_prec_score_macro}")
    print(f"Model precision (weighted): {avg_prec_score_weighted}")
    #print(f"Model precision (samples): {avg_prec_score_samples}")

    avg_recall_score_micro = recall_score(y_pred, y_test, average='micro')
    avg_recall_score_macro = recall_score(y_pred, y_test, average='macro')
    avg_recall_score_weighted = recall_score(y_pred, y_test, average='weighted')
    #avg_recall_score_samples = recall_score(y_pred, y_test, average='samples')
    avg_recall_score_overall = (avg_recall_score_micro + avg_recall_score_macro + avg_recall_score_weighted ) / 3

    print(f"Model recall (micro): {avg_recall_score_micro}")
    print(f"Model recall (macro): {avg_recall_score_macro}")
    print(f"Model recall (weighted): {avg_recall_score_weighted}")
    #print(f"Model recall (samples): {avg_recall_score_samples}")

    accuracy = model.score(X_test, y_test)
    return accuracy, avg_prec_score_overall, avg_recall_score_overall

