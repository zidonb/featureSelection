# data_preparation.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import os

from config import (
    RANDOM_STATE,
    SAMPLE_FRACTION,
    APPLY_FEATURE_FILTERING,
    APPLY_SCALING_ENCODING,
    OUTPUT_DIR
)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(path_X, path_y):
    """
    Load feature matrix (X) and labels (y) from disk.
    Assumes CSV format. Adjust if using another format.
    """
    X = pd.read_csv(path_X)
    y = pd.read_csv(path_y).values.ravel()  # Flatten to 1D array
    return X, y

def filter_features(X):
    """
    Optionally remove low-variance and highly correlated features.
    """
    # Remove low-variance features
    selector = VarianceThreshold(threshold=0.01)
    X_reduced = selector.fit_transform(X)
    selected_columns = X.columns[selector.get_support(indices=True)]
    X = pd.DataFrame(X_reduced, columns=selected_columns)

    # Remove highly correlated features
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    X.drop(columns=to_drop, inplace=True)

    return X

def scale_features(X):
    """
    Optionally scale features using StandardScaler.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns)

def stratified_sample(X, y, fraction):
    """
    Return a stratified sample of X and y.
    Ensures the class distribution remains consistent.
    """
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=1 - fraction, random_state=RANDOM_STATE)
    for train_idx, _ in splitter.split(X, y):
        return X.iloc[train_idx], y[train_idx]

def prepare_data(path_X, path_y):
    """
    Full data preparation pipeline:
    - Load
    - Optional filter
    - Optional scale
    - Stratified sample
    - Train/validation split
    
    Returns:
        X_train, X_val, y_train, y_val, feature_names
    """
    X, y = load_data(path_X, path_y)

    if APPLY_FEATURE_FILTERING:
        X = filter_features(X)

    if APPLY_SCALING_ENCODING:
        X = scale_features(X)

    X_sampled, y_sampled = stratified_sample(X, y, SAMPLE_FRACTION)

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_sampled, y_sampled, test_size=0.2, stratify=y_sampled, random_state=RANDOM_STATE
    )

    return X_train, X_val, y_train, y_val, X.columns.tolist()
