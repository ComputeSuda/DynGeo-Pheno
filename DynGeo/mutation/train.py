import os
import json
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.model_selection import StratifiedKFold

from model.model import build_seq_pssm_feature2_model
from utils.loss import focal_loss
from utils.metrics import find_optimal_threshold, cal_two_class_metric
from utils.data import set_random_seed, load_dataset_npy, split_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

set_random_seed(42)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

if len(gpus) >= 2:
    strategy = tf.distribute.MirroredStrategy()
else:
    strategy = tf.distribute.get_strategy()

GLOBAL_BATCH_SIZE = 48 * max(1, len(gpus))

if __name__ == "__main__":
    BASE_DIR = "."
    DATA_FILE = os.path.join(BASE_DIR, "dataset", "train_dataset.npz")
    MODEL_DIR = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load packaged dataset
    data = load_dataset_npy(DATA_FILE)
    train_label = data['label']
    dim1 = data['seq1'].shape[1]

    model_path = os.path.join(MODEL_DIR, "model_")
    cv_output = os.path.join(BASE_DIR, "cross_val_results.xlsx")

    # 2. 10-Fold Cross Validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_metrics = []
    all_val_preds = np.zeros(len(train_label))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(data['seq1'], train_label)):
        print(f"\n{'='*60}\nFold {fold+1}/10  train={len(tr_idx)}  val={len(va_idx)}\n{'='*60}")

        K.clear_session()
        tf.compat.v1.reset_default_graph()
        gc.collect()

        tr_inputs, va_inputs, y_tr, y_va = split_data(data, tr_idx, va_idx)

        with strategy.scope():
            model = build_seq_pssm_feature2_model(dim1)
            model.compile(
                optimizer=tf.keras.optimizers.RMSprop(learning_rate=9e-05, epsilon=1e-08),
                loss=focal_loss(gamma=2.0, alpha=0.55),
                metrics=['acc', tf.keras.metrics.AUC(name='auc')]
            )

        if fold == 0:
            model.summary()

        cbs = [
            tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=7, mode='max', restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=3, mode='max', min_lr=1e-6)
        ]

        model.fit(tr_inputs, y_tr, epochs=80, batch_size=GLOBAL_BATCH_SIZE, verbose=2,
                  validation_data=(va_inputs, y_va), callbacks=cbs)

        model.save(os.path.join(MODEL_DIR, f"model_{fold}.keras"), include_optimizer=False)

        y_pred = model.predict(va_inputs).flatten()
        all_val_preds[va_idx] = y_pred

        met = cal_two_class_metric(y_va, y_pred, threshold=0.5)
        met['fold'] = fold + 1
        fold_metrics.append(met)
        print(f"Fold {fold+1}: AUROC={met['auroc']:.4f}, F1={met['f1_score']:.4f}")

        del model
        K.clear_session()
        gc.collect()

    # 3. CV Summary & Threshold Search
    pd.DataFrame(fold_metrics).to_excel(cv_output, index=False)
    optimal_th, best_oof_f1, _ = find_optimal_threshold(train_label, all_val_preds, start=0.3, end=0.7, step=0.01)
    oof_metric = cal_two_class_metric(train_label, all_val_preds, threshold=optimal_th)

    print(f"\n{'='*60}\n10-Fold CV Complete\nOptimal threshold: {optimal_th:.2f}, F1: {best_oof_f1:.4f}\n{'='*60}")

    # 4. Save train_summary.json
    summary = {
        "optimal_threshold": float(optimal_th),
        "AUROC": float(oof_metric['auroc']),
        "AUPRC": float(oof_metric['auprc']),
        "F1": float(oof_metric['f1_score']),
        "Precision": float(oof_metric['precision']),
        "Recall": float(oof_metric['recall'])
    }
    summary_path = os.path.join(MODEL_DIR, "train_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nTraining complete. Summary saved to {summary_path}")
