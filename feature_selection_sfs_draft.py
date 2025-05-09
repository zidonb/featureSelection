# feature_selection_simple.py

import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
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
# Model and SFS
# -------------------------

model = MODEL  # Taken from config.py

sfs = SequentialFeatureSelector(
    model,
    n_features_to_select=NUM_FEATURES_TO_SELECT,
    direction='forward',
    scoring=SCORING,
    cv=3,
    n_jobs=-1
)

print("Fitting Sequential Feature Selector... This may take a while.")
sfs.fit(X_train, y_train)

# -------------------------
# Evaluate
# -------------------------

selected_features = X_train.columns[sfs.get_support()].tolist()
print("\nSelected Features:")
for feat in selected_features:
    print(feat)

# Optional: Test accuracy on validation set
X_val_selected = X_val[selected_features]
model.fit(X_train[selected_features], y_train)
y_pred = model.predict(X_val_selected)
val_score = accuracy_score(y_val, y_pred)
print(f"\nValidation {SCORING}: {val_score:.4f}")

# Save validation score and selected features to file
with open(SELECTED_FEATURES_PATH_SFS, 'w') as f:
    f.write(f"Validation {SCORING}: {val_score:.4f}\n")
    f.write("Selected Features:\n")
    for feat in selected_features:
        f.write(f"{feat}\n")
