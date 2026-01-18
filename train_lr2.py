import numpy as np
import pandas as pd
from logistic_regression import LogisticRegressionScratch, accuracy, roc_auc

SAUCES = [
    "Crazy Sauce",
    "Cheddar Sauce",
    "Extra Cheddar Sauce",
    "Garlic Sauce",
    "Tomato Sauce",
    "Blueberry Sauce",
    "Spicy Sauce",
    "Pink Sauce",
]

df = pd.read_csv("data/dataset_lr2.csv")
print(f"Dataset: {len(df)} receipts\n")

label_cols = [f"label_{s.replace(' ', '_').lower()}" for s in SAUCES]
feature_cols = [c for c in df.columns if c not in ["id_bon"] + label_cols]

X = df[feature_cols].values
mean, std = X.mean(axis=0), X.std(axis=0)
std[std == 0] = 1
X_norm = (X - mean) / std

# Train/test split
np.random.seed(42)
indices = np.random.permutation(len(X))
split = int(0.8 * len(X))
train_idx, test_idx = indices[:split], indices[split:]

# Train one model per sauce
models = {}
print("=== Training Models ===")
for sauce in SAUCES:
    label_col = f"label_{sauce.replace(' ', '_').lower()}"
    y = df[label_col].values

    model = LogisticRegressionScratch(
        learning_rate=0.1, n_iterations=500, regularization=0.01
    )
    model.fit(X_norm[train_idx], y[train_idx], verbose=False)
    models[sauce] = model

    y_pred = model.predict(X_norm[test_idx])
    y_proba = model.predict_proba(X_norm[test_idx])
    acc = accuracy(y[test_idx], y_pred)
    auc = roc_auc(y[test_idx], y_proba)
    print(f"{sauce}: Accuracy={acc:.3f}, AUC={auc:.3f}")

# Recommendation evaluation
print("\n=== Recommendation Evaluation ===")


def get_recommendations(x_sample, current_sauces, models, top_k=3):
    probs = {}
    for sauce, model in models.items():
        if sauce not in current_sauces:
            probs[sauce] = model.predict_proba(x_sample.reshape(1, -1))[0]
    return sorted(probs.items(), key=lambda x: x[1], reverse=True)[:top_k]


# Popularity baseline
sauce_popularity = {
    s: df[f"label_{s.replace(' ', '_').lower()}"].mean() for s in SAUCES
}
popularity_ranking = sorted(sauce_popularity.items(), key=lambda x: x[1], reverse=True)
print(f"\nPopularity baseline: {[s for s, _ in popularity_ranking[:3]]}")


# Evaluate Hit@K on test set
def evaluate_recommendations(test_indices, k_values=[1, 3, 5]):
    results = {k: {"model": 0, "baseline": 0, "total": 0} for k in k_values}

    for idx in test_indices:
        actual_sauces = [
            s
            for s in SAUCES
            if df.iloc[idx][f"label_{s.replace(' ', '_').lower()}"] == 1
        ]
        if not actual_sauces:
            continue

        x_sample = X_norm[idx]

        for k in k_values:
            results[k]["total"] += 1

            # Model recommendations
            recs = get_recommendations(x_sample, [], models, top_k=k)
            rec_sauces = [s for s, _ in recs]
            if any(s in rec_sauces for s in actual_sauces):
                results[k]["model"] += 1

            baseline_recs = [s for s, _ in popularity_ranking[:k]]
            if any(s in baseline_recs for s in actual_sauces):
                results[k]["baseline"] += 1

    return results


results = evaluate_recommendations(test_idx)

print("\nHit@K Results:")
print(f"{'K':<5} {'Model':<15} {'Baseline':<15}")
print("-" * 35)
for k in [1, 3, 5]:
    total = results[k]["total"]
    model_hit = results[k]["model"] / total if total > 0 else 0
    baseline_hit = results[k]["baseline"] / total if total > 0 else 0
    print(f"{k:<5} {model_hit:.3f}           {baseline_hit:.3f}")

# Example recommendation
print("\n=== Example Recommendation ===")
sample_idx = test_idx[0]
sample_sauces = [
    s
    for s in SAUCES
    if df.iloc[sample_idx][f"label_{s.replace(' ', '_').lower()}"] == 1
]
print(f"Actual sauces in cart: {sample_sauces}")
recs = get_recommendations(X_norm[sample_idx], [], models, top_k=3)
print(f"Model recommends: {[(s, f'{p:.2f}') for s, p in recs]}")
