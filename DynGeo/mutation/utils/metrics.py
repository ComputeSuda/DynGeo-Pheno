import numpy as np
import pandas as pd
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, precision_recall_curve, auc, confusion_matrix)

def find_optimal_threshold(y_true, y_pred_prob, start=0.3, end=0.7, step=0.01):
    best_threshold = 0.5
    best_f1 = 0
    thresholds = np.arange(start, end + step / 2, step)
    records = []

    for th in thresholds:
        y_pred = (y_pred_prob >= th).astype(int)
        current_f1 = f1_score(y_true, y_pred, zero_division=0)
        current_prec = precision_score(y_true, y_pred, zero_division=0)
        current_rec = recall_score(y_true, y_pred, zero_division=0)
        records.append({
            'threshold': round(th, 4),
            'f1': current_f1,
            'precision': current_prec,
            'recall': current_rec
        })
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = round(th, 4)

    df_search = pd.DataFrame(records)
    return best_threshold, best_f1, df_search

def cal_two_class_metric(y_true, y_pred_prob, threshold=0.5):
    y_true = np.array(y_true).flatten()
    y_pred_prob = np.array(y_pred_prob).flatten()
    y_pred = (y_pred_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    auroc = roc_auc_score(y_true, y_pred_prob) if len(np.unique(y_true)) > 1 else 0.0

    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_pred_prob)
    auprc = auc(rec_curve, prec_curve)

    metric = {
        'accuracy': accuracy, 'precision': precision, 'recall': recall,
        'f1_score': f1, 'auroc': auroc, 'auprc': auprc,
        'threshold': threshold,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }
    return metric
