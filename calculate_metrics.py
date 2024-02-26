from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
import numpy as np


def calculate_metrics(y_pred, y_true):
    """ Ground truth """
    
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    """ Prediction """
    y_pred = y_pred.cpu().detach().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_jaccard = jaccard_score(y_true, y_pred)
#     score_f1 = f1_score(y_true, y_pred)
#     score_recall = recall_score(y_true, y_pred)
#     score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_acc]