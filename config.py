# config.py

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

# -------------------------
# Feature Selection Config
# -------------------------

# Total number of top features to select
NUM_FEATURES_TO_SELECT = 55

# Evaluation metric
SCORING = 'accuracy'  # Or use 'f1_macro', 'balanced_accuracy', etc.

# Use stratified sample of data for faster training
SAMPLE_FRACTION = 0.05  # Use 5% of data during early iterations

# Random seed for reproducibility
RANDOM_STATE = 42

# -------------------------
# Classifier Configuration
# -------------------------

# Choose one model to use for feature scoring
# You can comment/uncomment to switch easily

#MODEL = LogisticRegression(solver='saga', multi_class='multinomial', max_iter=100, random_state=RANDOM_STATE)
#MODEL = RandomForestClassifier(n_estimators=20, n_jobs=-1, random_state=RANDOM_STATE)
MODEL = LGBMClassifier(objective='multiclass', n_estimators=50, n_jobs=-1, random_state=RANDOM_STATE)

# -------------------------
# Data File Paths
# -------------------------

# Path to feature matrix (CSV)
DATA_PATH_X = "data/features.csv"

# Path to label vector (CSV)
DATA_PATH_Y = "data/targets.csv"


# -------------------------
# Paths and Logging
# -------------------------

# Output folder for logs and selected features
OUTPUT_DIR = "outputs"

# Path for logs per iteration (scores, time, etc.)
ITERATION_LOG_PATH = f"{OUTPUT_DIR}/selection_scores.csv"

# Path to save final selected features
SELECTED_FEATURES_PATH = f"{OUTPUT_DIR}/selected_features.txt"

# Path to save confusion matrices (if applicable)
SELECTED_FEATURES_PATH_SFS = f"{OUTPUT_DIR}/selected_features_sfs.txt"
# Optional: save checkpoints after every N iterations
CHECKPOINT_EVERY = 5

# Whether to apply low-variance and high-correlation filtering
APPLY_FEATURE_FILTERING = True

# Whether to apply encoding/scaling (recommended for linear models)
APPLY_SCALING_ENCODING = False

