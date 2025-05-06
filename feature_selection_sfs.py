# feature_selection_sfs.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from config import DATA_PATH_X, DATA_PATH_Y, NUM_FEATURES_TO_SELECT, SCORING, RANDOM_STATE, SELECTED_FEATURES_PATH_SFS, MODEL

# -------------------------
# Load Data
# -------------------------
X = pd.read_csv(DATA_PATH_X)
y = pd.read_csv(DATA_PATH_Y).values.ravel()

# -------------------------
# Train/Validation Split
# -------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# -------------------------
# Greedy Forward Feature Selection (Manual)
# -------------------------
selected = []
remaining = list(X.columns)

print("\nStarting fast greedy forward selection (no CV)...")

for i in range(NUM_FEATURES_TO_SELECT):
    best_score = -1
    best_feature = None

    for feature in remaining:
        candidate_features = selected + [feature]
        MODEL.fit(X_train[candidate_features], y_train)
        preds = MODEL.predict(X_val[candidate_features])
        score = accuracy_score(y_val, preds)

        if score > best_score:
            best_score = score
            best_feature = feature

    if best_feature is None:
        print(f"No further improvement at iteration {i+1}. Stopping early.")
        break

    selected.append(best_feature)
    remaining.remove(best_feature)
    print(f"[{i+1:02d}] Selected: {best_feature} | Validation {SCORING}: {best_score:.4f}")

# Final model accuracy
MODEL.fit(X_train[selected], y_train)
final_preds = MODEL.predict(X_val[selected])
val_score = accuracy_score(y_val, final_preds)
print(f"\nFinal Validation {SCORING}: {val_score:.4f}")

# Save to file
with open(SELECTED_FEATURES_PATH_SFS, 'w') as f:
    f.write(f"Validation {SCORING}: {val_score:.4f}\n")
    f.write("Selected Features:\n")
    for feat in selected:
        f.write(f"{feat}\n")
