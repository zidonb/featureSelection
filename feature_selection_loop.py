# feature_selection_loop.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.metrics import confusion_matrix

from config import (
    NUM_FEATURES_TO_SELECT,
    RANDOM_STATE,
    ITERATION_LOG_PATH,
    SELECTED_FEATURES_PATH,
    OUTPUT_DIR,
    CHECKPOINT_EVERY,
    ENABLE_RECOVERY  # <-- Added for recovery control
)
from data_preparation import prepare_full_data
from feature_evaluation import evaluate_feature

# ---------------------------------------------
# Helper: Stratified sampling from full dataset
# ---------------------------------------------
def stratified_sample(X, y, fraction):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=1 - fraction, random_state=RANDOM_STATE)
    for train_idx, _ in splitter.split(X, y):
        return X.iloc[train_idx], y[train_idx]

# ---------------------------------------------
# Main Feature Selection Loop
# ---------------------------------------------
def run_feature_selection(path_X, path_y):
    # Load and preprocess full data
    X_full, y_full = prepare_full_data(path_X, path_y)
    all_features = X_full.columns.tolist()
    selected_features = []
    remaining_features = all_features.copy()

    # -------------------------
    # Recovery logic (NEW)
    # -------------------------
    if ENABLE_RECOVERY and os.path.exists(SELECTED_FEATURES_PATH):
        with open(SELECTED_FEATURES_PATH, 'r') as f:
            lines = f.read().splitlines()
        selected_features = [line.strip() for line in lines if line.strip()]
        remaining_features = [f for f in all_features if f not in selected_features]
        print(f"[Recovery] Loaded {len(selected_features)} previously selected features. Resuming from there.")

    # Prepare CSV log file
    log_columns = ["iteration", "feature", "score", "duration_sec", "conf_matrix_path"]
    if not os.path.exists(ITERATION_LOG_PATH):
        pd.DataFrame(columns=log_columns).to_csv(ITERATION_LOG_PATH, index=False)

    # Create output dir for confusion matrices
    cm_dir = os.path.join(OUTPUT_DIR, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)

    # -------------------------
    # Determine starting iteration (NEW)
    # -------------------------
    start_iteration = len(selected_features) + 1

    for iteration in range(start_iteration, NUM_FEATURES_TO_SELECT + 1):
        print(f"\n[Iteration {iteration}] Selecting best feature...")

        # Dynamically adjust sampling fraction
        sample_fraction = min(0.05 + 0.01 * iteration, 1.0)
        X_sample, y_sample = stratified_sample(X_full, y_full, sample_fraction)

        # Split sample into train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_sample, y_sample, test_size=0.2, stratify=y_sample, random_state=RANDOM_STATE
        )

        iteration_results = []
        best_score = -np.inf
        best_feature = None
        best_matrix = None
        best_duration = None

        for feature in remaining_features:
            score, conf_matrix, duration = evaluate_feature(
                X_train, y_train, X_val, y_val, selected_features, feature
            )
            iteration_results.append({
                "iteration": iteration,
                "feature": feature,
                "score": score,
                "duration_sec": duration,
                "conf_matrix_path": None  # default; only updated for best
            })

            # Track best feature in this iteration
            if score > best_score:
                best_score = score
                best_feature = feature
                best_matrix = conf_matrix
                best_duration = duration

        # Save best confusion matrix
        cm_filename = f"iter_{iteration:02d}_feat_{best_feature}.csv"
        cm_path = os.path.join(cm_dir, cm_filename)
        pd.DataFrame(best_matrix).to_csv(cm_path, index=False)

        # Update corresponding row with cm path
        for row in iteration_results:
            if row["feature"] == best_feature:
                row["conf_matrix_path"] = cm_path

        # Append all rows from this iteration to the log file
        pd.DataFrame(iteration_results).to_csv(ITERATION_LOG_PATH, mode='a', header=False, index=False)

        # Update selected features
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)

        print(f"[✓] Selected: {best_feature}  | Score: {best_score:.4f}")

        # Checkpointing
        if iteration % CHECKPOINT_EVERY == 0 or iteration == NUM_FEATURES_TO_SELECT:
            with open(SELECTED_FEATURES_PATH, 'w') as f:
                for feat in selected_features:
                    f.write(f"{feat}\n")
