import os
import json
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import models

from model.model import (build_seq_pssm_feature2_model, InverseLayer, 
                         GraphConvolutionLayer, GraphAttentionLayer, 
                         GAT, GCN_GATLayer)
from utils.metrics import cal_two_class_metric
from utils.data import set_random_seed, load_dataset_npy, get_test_inputs

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

set_random_seed(42)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

CUSTOM_OBJECTS = {
    'GCN_GATLayer': GCN_GATLayer,
    'GAT': GAT,
    'GraphAttentionLayer': GraphAttentionLayer,
    'GraphConvolutionLayer': GraphConvolutionLayer,
    'InverseLayer': InverseLayer
}

def test_ensemble_model(test_inputs, test_label, model_dir, optimal_threshold, output_filename):
    """Load 10-fold models for ensemble prediction."""
    ensemble_pred = np.zeros(len(test_label))
    single_model_metrics = []

    print(f"\n===== 10-Fold Ensemble Test =====")
    for i in range(10):
        model_file = os.path.join(model_dir, f"model_{i}.keras")
        if not os.path.exists(model_file):
            print(f"Model {model_file} not found, skipping")
            continue

        K.clear_session()
        gc.collect()

        model = tf.keras.models.load_model(model_file, compile=False, custom_objects=CUSTOM_OBJECTS)
        pred = model.predict(test_inputs, verbose=0).flatten()
        ensemble_pred += pred

        single_metric = cal_two_class_metric(test_label, pred, threshold=optimal_threshold)
        single_metric['model_id'] = i
        single_model_metrics.append(single_metric)
        print(f"Model {i}: AUROC={single_metric['auroc']:.4f}, F1={single_metric['f1_score']:.4f}")

    n_loaded = len(single_model_metrics)
    if n_loaded == 0:
        print("Error: No models loaded!")
        return

    ensemble_pred /= n_loaded

    test_metric = cal_two_class_metric(test_label, ensemble_pred, threshold=optimal_threshold)
    test_metric_default = cal_two_class_metric(test_label, ensemble_pred, threshold=0.5)

    print("\n" + "=" * 60)
    print(f"Ensemble Test Results ({n_loaded} models)")
    print(f"-- OOF optimal threshold {optimal_threshold:.2f} --")
    print(f"  Accuracy :  {test_metric['accuracy']:.4f}")
    print(f"  Precision:  {test_metric['precision']:.4f}")
    print(f"  Recall   :  {test_metric['recall']:.4f}")
    print(f"  F1       :  {test_metric['f1_score']:.4f}")
    print(f"  AUROC    :  {test_metric['auroc']:.4f}")
    print(f"  AUPRC    :  {test_metric['auprc']:.4f}")
    print(f"-- Default threshold 0.50 --")
    print(f"  Precision:  {test_metric_default['precision']:.4f}")
    print(f"  Recall   :  {test_metric_default['recall']:.4f}")
    print(f"  F1       :  {test_metric_default['f1_score']:.4f}")
    print("=" * 60)

    test_metric['ensemble_n_models'] = n_loaded
    df_result = pd.DataFrame([test_metric])
    df_result.to_excel(output_filename, index=False)

    single_output = output_filename.replace(".xlsx", "_single_model_compare.xlsx")
    pd.DataFrame(single_model_metrics).to_excel(single_output, index=False)

    return test_metric


if __name__ == "__main__":
    BASE_DIR = "."
    DATA_FILE = os.path.join(BASE_DIR, "dataset", "test_dataset.npz")
    MODEL_DIR = os.path.join(BASE_DIR, "checkpoints")
    OUTPUT_FILE = os.path.join(BASE_DIR, "test_results.xlsx")

    # 1. Load optimal threshold from train_summary.json
    summary_path = os.path.join(MODEL_DIR, "train_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError("Missing train_summary.json. Please run train.py first.")
    
    with open(summary_path, "r") as f:
        optimal_threshold = json.load(f)["optimal_threshold"]
    print(f"Loaded optimal threshold: {optimal_threshold}")

    # 2. Load test dataset
    data = load_dataset_npy(DATA_FILE)
    test_label = data['label']
    test_inputs = get_test_inputs(data)

    # 3. Run ensemble test
    test_ensemble_model(
        test_inputs, 
        test_label, 
        MODEL_DIR, 
        optimal_threshold, 
        OUTPUT_FILE
    )
    
    print(f"\nTesting complete. Results saved to {OUTPUT_FILE}")
