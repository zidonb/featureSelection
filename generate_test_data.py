# generate_test_data.py

from sklearn.datasets import load_digits
import pandas as pd

# Load the digits dataset (multi-class classification)
digits = load_digits()

# Create DataFrames for features (X) and labels (y)
X = pd.DataFrame(digits.data)
y = pd.DataFrame(digits.target, columns=['label'])

# Save to CSV files
X.to_csv("your_X.csv", index=False)
y.to_csv("your_y.csv", index=False)

print("Test data generated and saved as 'your_X.csv' and 'your_y.csv'")
