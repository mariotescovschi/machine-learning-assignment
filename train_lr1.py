import numpy as np
import pandas as pd
from logistic_regression import (
    LogisticRegressionScratch,
    accuracy,
    confusion_matrix,
    precision_recall_f1,
    roc_auc,
)

df = pd.read_csv("data/dataset_lr1.csv")
print(f"Dataset: {len(df)} receipts with Crazy Schnitzel")
print(f"Label distribution: {df['label'].value_counts().to_dict()}")

feature_cols = [c for c in df.columns if c not in ["id_bon", "label"]]
X = df[feature_cols].values
y = df["label"].values

# Train/test split
np.random.seed(42)
indices = np.random.permutation(len(X))
split = int(0.8 * len(X))
train_idx, test_idx = indices[:split], indices[split:]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Normalize
mean, std = X_train.mean(axis=0), X_train.std(axis=0)
std[std == 0] = 1
X_train_norm = (X_train - mean) / std
X_test_norm = (X_test - mean) / std

# Train
print("\n--- Training ---")
model = LogisticRegressionScratch(
    learning_rate=0.1, n_iterations=1000, regularization=0.01
)
model.fit(X_train_norm, y_train)

# Evaluate
print("\n--- Test Results ---")
y_pred = model.predict(X_test_norm)
y_proba = model.predict_proba(X_test_norm)

print(f"Accuracy: {accuracy(y_test, y_pred):.4f}")
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix: TN={cm[0, 0]}, FP={cm[0, 1]}, FN={cm[1, 0]}, TP={cm[1, 1]}")

p, r, f1 = precision_recall_f1(y_test, y_pred)
print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
print(f"ROC-AUC: {roc_auc(y_test, y_proba):.4f}")

# Baseline
majority = 1 if y_train.sum() > len(y_train) / 2 else 0
print(
    f"\nBaseline (majority={majority}): {accuracy(y_test, np.full_like(y_test, majority)):.4f}"
)

# Coefficients
print("\n--- Feature Weights ---")
coef_df = pd.DataFrame({"feature": feature_cols, "weight": model.weights})
coef_df = coef_df.sort_values("weight", ascending=False)
print("\nPositive (increase Crazy Sauce probability):")
print(coef_df.head(10).to_string(index=False))
print("\nNegative (decrease probability):")
print(coef_df.tail(10).to_string(index=False))
