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
        seq1=data['esm'],
        cmap=data['cmap'],
        kf_f=data['anm_front'],
        kf_l=data['anm_last']
    )
    
    print(f"   Samples  : {len(dataset['label'])}")
    print(f"   ESM      : {dataset['seq1'].shape}")
    print(f"   ANM_front: {dataset['kf_f'].shape}")
    print(f"   ANM_last : {dataset['kf_l'].shape}")
    print(f"   CMap     : {dataset['cmap'].shape}")
    return dataset

def split_data(data, tr_idx, va_idx):
    """Split dataset according to cross-validation indices."""
    keys = ['seq1', 'kf_f', 'kf_l', 'cmap']
    return ([data[k][tr_idx] for k in keys],
            [data[k][va_idx] for k in keys],
            data['label'][tr_idx], data['label'][va_idx])

def get_test_inputs(data):
    """Convert feature dictionary to input list required by model predict."""
    keys = ['seq1', 'kf_f', 'kf_l', 'cmap']
    return [data[k] for k in keys]
