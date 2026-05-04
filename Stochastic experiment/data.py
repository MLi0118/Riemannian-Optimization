"""
UCI Superconductivity loader.

The PL trigger requires |D_tr| < d (lower-level dimension). We pick a small
training set (m_tr = 30) and reserve the rest as a large validation set so
that mini-batch sweeps over D_val are also meaningful.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


def load_superconductivity(csv_path, m_tr=30, m_val=20000, seed=0,
                           dtype=torch.float64, device='cpu'):
    """Load UCI Superconductivity, standardise, split into (D_tr, D_val).

    Returns (X_tr, y_tr, X_val, y_val) as torch tensors. Features and target
    are standardised using D_tr statistics only (no leakage).
    """
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-1].values.astype(np.float64)   # (21263, 81)
    y = df.iloc[:, -1].values.astype(np.float64)    # critical_temp

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    tr_idx = perm[:m_tr]
    val_idx = perm[m_tr:m_tr + m_val]

    # Standardise features on D_tr, apply to D_val
    feat_scaler = StandardScaler().fit(X[tr_idx])
    X_tr = feat_scaler.transform(X[tr_idx])
    X_val = feat_scaler.transform(X[val_idx])

    # Standardise target on D_tr
    y_mean = y[tr_idx].mean()
    y_std = y[tr_idx].std() + 1e-12
    y_tr = (y[tr_idx] - y_mean) / y_std
    y_val = (y[val_idx] - y_mean) / y_std

    # Additionally normalise feature row norms so ||x_i|| ~ 1; keeps gradient
    # magnitudes comparable across manifolds without retuning stepsizes.
    X_tr = X_tr / np.sqrt(X.shape[1])
    X_val = X_val / np.sqrt(X.shape[1])

    return (torch.tensor(X_tr,  dtype=dtype, device=device),
            torch.tensor(y_tr,  dtype=dtype, device=device),
            torch.tensor(X_val, dtype=dtype, device=device),
            torch.tensor(y_val, dtype=dtype, device=device))
