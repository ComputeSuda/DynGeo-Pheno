import os
import json
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import models

from model.model import get_custom_objects
from utils.metrics import cal_two_class_metric
from utils.data import set_random_seed, load_test_data, get_test_inputs

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SEED = 42
set_random_seed(SEED)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

NUM_GPUS = len(gpus)
BATCH_SIZE = 16 * max(1, NUM_GPUS)


def adaptive_quality_ensemble(test_inputs, test_label, model_dir, threshold,
                              n_folds=10):
    custom_objects = get_custom_objects()
    fold_preds = {}
    fold_aucs = {}
    fold_metrics = []

    print(f"\n===== 10-Fold Ensemble Test =====")
    for fold in range(n_folds):
        model_path = os.path.join(model_dir, f'model_{fold}.keras')
        if not os.path.exists(model_path):
            print(f"   Fold {fold} model missing, skipping")
            continue

        K.clear_session()
        gc.collect()

        m = models.load_model(model_path, compile=False,
                              custom_objects=custom_objects)
        pred = m.predict(test_inputs, batch_size=BATCH_SIZE).flatten()

        fold_met = cal_two_class_metric(test_label, pred, threshold=threshold)
        fold_met['fold'] = fold
        fold_metrics.append(fold_met)

        fold_preds[fold] = pred
        fold_aucs[fold] = fold_met['auroc']

        print(f"   Fold {fold}: test_auc = {fold_met['auroc']:.4f}")
        del m

    if not fold_aucs:
        print("No models found!")
        return None

    # Adaptive quality ensemble: select folds with AUC above median
    auc_values = list(fold_aucs.values())
    auc_median = np.median(auc_values)

    selected_folds = [f for f, a in fold_aucs.items() if a > auc_median]
    if len(selected_folds) == 0:
        selected_folds = list(fold_aucs.keys())

    print(f"\nTest fold AUC median: {auc_median:.4f}")
    print(f"Selected folds for ensemble: {len(selected_folds)}/{len(fold_aucs)}")
    print(f"Selected: {selected_folds}")

    ensemble_pred = np.zeros(len(test_label))
    for fold in selected_folds:
        ensemble_pred += fold_preds[fold]
    ensemble_pred /= len(selected_folds)

    test_met = cal_two_class_metric(test_label, ensemble_pred, threshold=threshold)
    test_met['fold'] = 'ensemble'
    fold_metrics.append(test_met)

    print(f"\n{'='*60}")
    print(f"Ensemble Test Results (threshold={threshold:.2f})")
    for k, v in test_met.items():
        if k != 'fold':
            print(f"   {k:18s}: {v:.4f}")
    print(f"{'='*60}")

    return fold_metrics


if __name__ == "__main__":
    BASE_DIR  = "."
    FEAT_DIR  = "./example"
    MODEL_DIR = os.path.join(BASE_DIR, "checkpoints")
    SPLIT = 9

    # Load threshold from train_summary.json
    summary_path = os.path.join(MODEL_DIR, "train_summary.json")
    with open(summary_path, "r") as f:
        summary = json.load(f)
    threshold = summary["optimal_threshold"]
    print(f"Loaded optimal threshold from train_summary.json: {threshold}")

    # Load test data
    data = load_test_data(FEAT_DIR, SPLIT)
    test_label = data['label']
    test_inputs = get_test_inputs(data)

    # Run adaptive quality ensemble
    all_metrics = adaptive_quality_ensemble(
        test_inputs, test_label, MODEL_DIR, threshold, n_folds=10)

    if all_metrics is not None:
        df_test = pd.DataFrame(all_metrics)
        output_file = os.path.join(BASE_DIR, "test_results.xlsx")
        df_test.to_excel(output_file, index=False)
        print(f"\nResults saved to {output_file}")

    print("\nTesting complete.")
