import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


def find_optimal_threshold(y_true, y_pred_prob):
    y_true = np.array(y_true).flatten()
    y_pred_prob = np.array(y_pred_prob).flatten()
    best_f1, best_th = 0, 0.5
    for th in np.arange(0.3, 0.7, 0.01):
        y_pred = (y_pred_prob >= th).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        if f1 > best_f1:
            best_f1, best_th = f1, th
    return best_th


def cal_two_class_metric(y_true, y_pred_prob, threshold=None):
    y_true = np.array(y_true).flatten()
    y_pred_prob = np.array(y_pred_prob).flatten()
    if threshold is None:
        threshold = find_optimal_threshold(y_true, y_pred_prob)
    y_pred = (y_pred_prob >= threshold).astype(int)

    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))

    accuracy  = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2*precision*recall / (precision + recall) if (precision + recall) > 0 else 0.0
    auroc     = roc_auc_score(y_true, y_pred_prob) if len(np.unique(y_true)) > 1 else 0.0
    prec_c, rec_c, _ = precision_recall_curve(y_true, y_pred_prob)
    auprc     = auc(rec_c, prec_c)

    return dict(accuracy=accuracy, precision=precision, recall=recall,
                f1_score=f1, auroc=auroc, auprc=auprc,
                optimal_threshold=threshold)
