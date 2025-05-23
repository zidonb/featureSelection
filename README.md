# Greedy Forward Feature Selection for Multi-Class Classification

This project provides two feature selection pipelines for classifying companies into economic subcategories based on structured business data.

---

## 📦 Overview

We implement and compare:

### ✅ Full Greedy Selection (Custom)
- Iteratively selects features using greedy forward strategy
- Dynamic sampling to speed up training
- Logs score per feature, per iteration
- Saves confusion matrices for selected features
- Configurable filters (low-variance, correlation)
- Outputs:
  - `selection_scores.csv`
  - `confusion_matrices/`
  - `selected_features.txt`

### ✅ Simple SFS Version (scikit-learn)
- Uses `SequentialFeatureSelector` with any model
- Plug-and-play, config-driven
- Outputs:
  - `selected_features_sfs.txt`

---

## 📁 Structure

```plaintext
.
├── config.py                      # Central settings for model, paths, and strategy
├── main.py                        # Runs full greedy feature selection
├── feature_selection_loop.py      # Main loop for greedy feature selection
├── feature_evaluation.py          # Model training and evaluation logic
├── data_preparation.py            # Loads and optionally filters/scales data
├── feature_selection_sfs.py       # scikit-learn SFS-based alternative
├── generate_test_data.py          # Generates digits test dataset
├── outputs/                       # All results, logs, and selected features
│   ├── selection_scores.csv
│   ├── selected_features.txt
│   ├── selected_features_sfs.txt
│   └── confusion_matrices/
````

---

## 🧪 Test With Built-in Dataset

Generate a small test dataset using:

```bash
python generate_test_data.py
```

It creates:

* `data/features.csv`
* `data/targets.csv`

Update `config.py`:

```python
DATA_PATH_X = "data/features.csv"
DATA_PATH_Y = "data/targets.csv"
```

---

## 🚀 Run

### Full Version (greedy with logging):

```bash
python main.py
```

### Simple Version (SFS):

```bash
python feature_selection_sfs.py
```

---

## ⚙️ Configuration

Customize:

* Number of features to select
* Sampling rate
* Model (LightGBM recommended for speed)
* Filtering options
  All in `config.py`.

---

## 📌 Notes

* `SequentialFeatureSelector` can still be slow with CV and large feature counts
* For better performance, use tree-based models like LightGBM (`LGBMClassifier`)
* Both pipelines can be adapted for different data types or scoring methods

---

## 📄 License

MIT License
