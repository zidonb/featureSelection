# main.py

from feature_selection_loop import run_feature_selection
from config import DATA_PATH_X, DATA_PATH_Y

if __name__ == "__main__":
    print("Starting greedy forward feature selection...")
    run_feature_selection(DATA_PATH_X, DATA_PATH_Y)
    print("\n[✓] Feature selection completed. Results saved to output files.")
