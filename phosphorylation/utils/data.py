import os
import random
import numpy as np
import tensorflow as tf

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_dataset_npy(data_file):
    """Load all features and labels from a packaged .npz file."""
    print(f"\nLoading data from {data_file} ...")
    data = np.load(data_file)
    
    dataset = dict(
        label=data['labels'],
        knn_esm=data['knn_esm'],
        knn_kf_f=data['knn_anm_front'],
        knn_kf_l=data['knn_anm_last'],
        knn_cmap=data['knn_cmap'],
        p1_esm=data['p1_esm'],
        p1_kf_f=data['p1_anm_front'],
        p1_kf_l=data['p1_anm_last'],
        p1_cmap=data['p1_cmap']
    )
    
    print(f"   Samples  : {len(dataset['label'])}")
    return dataset

def split_data(data, tr_idx, va_idx):
    """Split dataset according to cross-validation indices."""
    keys = ['knn_esm', 'knn_kf_f', 'knn_kf_l', 'knn_cmap', 'p1_esm', 'p1_kf_f', 'p1_kf_l', 'p1_cmap']
    return ([data[k][tr_idx] for k in keys],
            [data[k][va_idx] for k in keys],
            data['label'][tr_idx], data['label'][va_idx])

def get_test_inputs(data):
    """Convert feature dictionary to input list required by model predict."""
    keys = ['knn_esm', 'knn_kf_f', 'knn_kf_l', 'knn_cmap', 'p1_esm', 'p1_kf_f', 'p1_kf_l', 'p1_cmap']
    return [data[k] for k in keys]
