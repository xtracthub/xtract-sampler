
from sklearn.metrics import precision_score, recall_score, plot_confusion_matrix, plot_precision_recall_curve
from sklearn.metrics._plot.roc_curve import plot_roc_curve
import matplotlib.pyplot as plt

def score_model(model, X_test, Y_test):
    """Scores the model.

    Parameters:
    model (sklearn model): Trained sklearn model (SVC, LinearRegression,
    RandForestClassifier).
    X_test (list): List of files to test on.
    Y_test (list): List of labels for X_test.

    Return:
    (float): The percentage of files from X_test that model was able to
    correctly classify.
    """


    y_pred = model.predict(X_test)
    avg_prec_score_micro = precision_score(y_pred, Y_test, average='micro')
    avg_prec_score_macro = precision_score(y_pred, Y_test, average='macro')
    avg_prec_score_weighted = precision_score(y_pred, Y_test, average='weighted')
    #avg_prec_score_samples = precision_score(y_pred, y_test, average='samples')
    avg_prec_score_overall = (avg_prec_score_micro + avg_prec_score_macro + avg_prec_score_weighted) / 3

    print(f"Model precision (micro): {avg_prec_score_micro}")
    print(f"Model precision (macro): {avg_prec_score_macro}")
    print(f"Model precision (weighted): {avg_prec_score_weighted}")
    #print(f"Model precision (samples): {avg_prec_score_samples}")

    avg_recall_score_micro = recall_score(y_pred, Y_test, average='micro')
    avg_recall_score_macro = recall_score(y_pred, Y_test, average='macro')
    avg_recall_score_weighted = recall_score(y_pred, Y_test, average='weighted')
    #avg_recall_score_samples = recall_score(y_pred, y_test, average='samples')
    avg_recall_score_overall = (avg_recall_score_micro + avg_recall_score_macro + avg_recall_score_weighted ) / 3

    print(f"Model recall (micro): {avg_recall_score_micro}")
    print(f"Model recall (macro): {avg_recall_score_macro}")
    print(f"Model recall (weighted): {avg_recall_score_weighted}")
    #print(f"Model recall (samples): {avg_recall_score_samples}")

    accuracy = model.score(X_test, Y_test)
    print(f"Model accuracy: {accuracy}")

    # UNCOMMENT TO PRODUCE CONFUSION MATRIX
    ''' 
    disp = plot_confusion_matrix(model, X_test, Y_test,
                            display_labels=list(self.class_table.keys()),                                 cmap=plt.cm.Blues,
                            normalize=None)
    disp.ax_.set_title('SVC Confusion Matrix')
    plt.savefig('SVC Confusion Matrix No Normalize.png', format='png')
    '''

    # UNCOMMENT TO MAKE ROC CURVE
    '''
    disp = plot_roc_curve(model, X_test, Y_test)
    disp.ax_.set_title('ROC Curve')
    plt.savefig(' ROC Curve', format='png')
    '''

    # UNCOMMENT TO MAKE Precision Recall CURVE
    '''
    disp = plot_precision_recall_curve(model, X_test, Y_test)
    disp.ax_.set_title(' PR Curve')
    plt.savefig(' PR Curve', format='png')
    '''
    

    return accuracy, avg_prec_score_overall, avg_recall_score_overall

