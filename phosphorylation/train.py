import os
import json
import gc
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.model_selection import StratifiedKFold

from model.model import build_gated_gcngat_model
from utils.loss import FocalLoss, create_optimizer
from utils.metrics import cal_two_class_metric
from utils.data import set_random_seed, load_training_data, split_data

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
print(f"Detected {NUM_GPUS} GPU(s)")

if NUM_GPUS >= 2:
    strategy = tf.distribute.MirroredStrategy()
    print(f"MirroredStrategy ({NUM_GPUS} GPU)")
else:
    strategy = tf.distribute.get_strategy()
    print("Single GPU mode")

BATCH_SIZE_PER_GPU = 16
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_GPU * max(1, NUM_GPUS)

CONFIG = {
    'nh':           218,
    'heads':        4,
    'dp':           0.3,
    'lamda':        1.3,
    'alpha':        0.1,
    'l':            5,
    'nlayers':      3,
    'lr':           2e-4,
    'weight_decay': 1e-5,
    'clipnorm':     1.0,
    'epochs':       40,
    'patience_es':  7,
    'patience_lr':  3,
}


if __name__ == "__main__":
    BASE_DIR  = "."
    FEAT_DIR  = "./example"
    MODEL_DIR = os.path.join(BASE_DIR, "checkpoints")
    LOG_DIR   = os.path.join(BASE_DIR, "logs")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    SPLIT = 9

    data = load_training_data(FEAT_DIR, SPLIT)
    total_label = data['label']
    dim1      = data['seq1'].shape[1]
    dim2      = data['p1esm'].shape[1]
    kf_dim    = data['kf_f'].shape[-1]
    p1_kf_dim = data['p1kf_f'].shape[-1]

    print(f"\nLoss: FocalLoss(gamma=2.0, alpha=0.5)")
    opt, opt_name = create_optimizer(CONFIG)
    print(f"Optimizer: {opt_name}  lr={CONFIG['lr']}  nh={CONFIG['nh']}  dp={CONFIG['dp']}")

    # --- 10-Fold Cross Validation ---
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    fold_metrics = []
    all_val_preds = np.zeros(len(total_label))

    for fold, (tr_idx, va_idx) in enumerate(
            skf.split(data['seq1'], total_label)):
        print(f"\n{'='*60}")
        print(f"Fold {fold+1}/10  train={len(tr_idx)}  val={len(va_idx)}")
        print(f"{'='*60}")
        t_fold = time.time()

        K.clear_session()
        tf.compat.v1.reset_default_graph()
        gc.collect()

        tr_inputs, va_inputs, y_tr, y_va = split_data(data, tr_idx, va_idx)

        with strategy.scope():
            model = build_gated_gcngat_model(
                dim1, dim2, kf_dim, p1_kf_dim, fold+1, CONFIG)
            opt, _ = create_optimizer(CONFIG)
            model.compile(
                optimizer=opt,
                loss=FocalLoss(gamma=2.0, alpha=0.5),
                metrics=['acc',
                         tf.keras.metrics.AUC(name='auc'),
                         tf.keras.metrics.AUC(curve='PR', name='auprc')])

        if fold == 0:
            model.summary()

        cbs = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc', patience=CONFIG['patience_es'],
                mode='max', restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_auc', factor=0.5, patience=CONFIG['patience_lr'],
                mode='max', min_lr=1e-6, verbose=1),
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(LOG_DIR, f'fold_{fold+1}'),
                update_freq='epoch'),
        ]

        hist = model.fit(
            tr_inputs, y_tr,
            epochs=CONFIG['epochs'],
            batch_size=GLOBAL_BATCH_SIZE,
            verbose=2,
            validation_data=(va_inputs, y_va),
            callbacks=cbs)

        model.save(os.path.join(MODEL_DIR, f'model_{fold}.keras'),
                   include_optimizer=False)

        y_pred_va = model.predict(va_inputs,
                                  batch_size=GLOBAL_BATCH_SIZE).flatten()
        all_val_preds[va_idx] = y_pred_va

        met = cal_two_class_metric(y_va, y_pred_va)
        met['fold'] = fold + 1
        fold_metrics.append(met)

        print(f"\nFold {fold+1} results ({time.time()-t_fold:.0f}s):")
        for k, v in met.items():
            if k != 'fold':
                print(f"   {k:18s}: {v:.4f}")

        del model, hist
        K.clear_session()
        gc.collect()

    # --- CV Summary ---
    df_cv = pd.DataFrame(fold_metrics)
    oof_met = cal_two_class_metric(total_label, all_val_preds)

    print(f"\n{'='*60}")
    print("10-Fold CV Complete")
    print(f"\nPer-fold metrics:")
    print(df_cv.to_string(index=False))
    print(f"\nOOF metrics:")
    for k, v in oof_met.items():
        print(f"   {k:18s}: {v:.4f}")

    df_cv.to_excel(os.path.join(BASE_DIR, 'cross_val_results.xlsx'),
                   index=False)

    # --- Save train_summary.json ---
    summary = {
        "optimal_threshold": float(oof_met['optimal_threshold']),
        "AUROC": float(oof_met['auroc']),
        "AUPRC": float(oof_met['auprc']),
        "F1": float(oof_met['f1_score']),
        "Precision": float(oof_met['precision']),
        "Recall": float(oof_met['recall']),
        "Accuracy": float(oof_met['accuracy']),
    }
    summary_path = os.path.join(MODEL_DIR, "train_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\nTraining complete. Summary saved to {summary_path}")
