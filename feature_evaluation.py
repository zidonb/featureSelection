# feature_evaluation.py

import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from config import MODEL, SCORING


def evaluate_feature(X_train, y_train, X_val, y_val, selected_features, candidate_feature):
    """
    Trains and evaluates the model using selected + candidate feature.

    Args:
        X_train (DataFrame): Training features
        y_train (array): Training labels
        X_val (DataFrame): Validation features
        y_val (array): Validation labels
        selected_features (list): Already selected feature names
        candidate_feature (str): Feature to test in this iteration

    Returns:
        score (float): Evaluation score based on SCORING method
        conf_matrix (ndarray): Confusion matrix of predictions
        duration (float): Time taken in seconds
    """
    start_time = time.time()

    # Combine selected + candidate features
    feature_set = selected_features + [candidate_feature]

    # Slice data accordingly
    X_train_sub = X_train[feature_set]
    X_val_sub = X_val[feature_set]

    # Train model
    model = MODEL
    model.fit(X_train_sub, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_val_sub)

    if SCORING == 'accuracy':
        score = accuracy_score(y_val, y_pred)
    elif SCORING == 'f1_macro':
        score = f1_score(y_val, y_pred, average='macro')
    else:
        raise ValueError(f"Unsupported scoring metric: {SCORING}")

    # Get confusion matrix
    conf_matrix = confusion_matrix(y_val, y_pred)

    duration = time.time() - start_time

    return score, conf_matrix, duration
