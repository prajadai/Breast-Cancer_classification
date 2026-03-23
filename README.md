# Breast Cancer Classification with a Decision Tree (From Scratch)

This project builds and evaluates a binary classification Decision Tree for breast cancer diagnosis using pure Python, NumPy, and pandas.

The notebook demonstrates:
- Data loading and quick inspection
- Basic cleaning and target encoding
- A custom Decision Tree implementation
- Stratified train/test split
- Feature importance estimation
- Evaluation with accuracy, classification report, and confusion matrix

## Project Files

- `breast-cancer-classification.ipynb` - Main notebook with the full workflow
- `data.csv` - Breast cancer dataset used by the notebook

## Requirements

- Python 3.9+
- Jupyter Notebook or VS Code Jupyter extension
- Python packages:
  - numpy
  - pandas

## Setup

1. Open a terminal in this folder.
2. (Optional) Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install numpy pandas notebook
```

## Run

1. Start Jupyter:

```bash
jupyter notebook
```

2. Open `breast-cancer-classification.ipynb`.
3. Run cells from top to bottom.

## Workflow Summary

1. Load the dataset.
2. Drop non-useful columns (`id`, `Unnamed: 32`).
3. Encode diagnosis into a binary target (`M -> 1`, `B -> 0`).
4. Train a custom Decision Tree class.
5. Evaluate with train/test accuracy, per-class metrics, and confusion matrix.

## Notes

- The notebook currently uses `pd.read_csv('/data.csv')`, which may fail on some systems because it is an absolute path.
- If that happens, change it to:

```python
df = pd.read_csv('data.csv')
```

## Learning Objective

This project is designed to help understand how Decision Trees work internally (splits, impurity, information gain, recursion, and predictions) rather than relying on a prebuilt model implementation.
