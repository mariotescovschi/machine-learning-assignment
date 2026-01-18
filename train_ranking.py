import json
import numpy as np
from collections import defaultdict
from naive_bayes import NaiveBayesRanking, PopularityBaseline

CANDIDATES = [
    "Crazy Sauce",
    "Cheddar Sauce",
    "Extra Cheddar Sauce",
    "Garlic Sauce",
    "Tomato Sauce",
    "Blueberry Sauce",
    "Spicy Sauce",
    "Pink Sauce",
    "Pepsi Cola 0.25L Doze",
    "Pepsi Zero Can 0.33L",
    "Mountain Dew 0.25L Doze",
    "Aqua Carpatica Minerala 0.5L",
    "French Fries",
    "Baked Potatoes",
]

with open("./data/receipts.json", "r") as f:
    receipts = json.load(f)

carts = []
prices = defaultdict(list)

for r in receipts:
    cart = []
    for item in r["items"]:
        name = item["retail_product_name"]
        cart.append(name)
        prices[name].append(item["SalePriceWithVAT"])
    carts.append(cart)

avg_prices = {p: np.mean(v) for p, v in prices.items()}

np.random.seed(42)
indices = np.random.permutation(len(carts))
split = int(0.8 * len(carts))
train_carts = [carts[i] for i in indices[:split]]
test_carts = [carts[i] for i in indices[split:]]

print(f"Train: {len(train_carts)}, Test: {len(test_carts)}")

nb_model = NaiveBayesRanking()
nb_model.fit(train_carts, avg_prices)

baseline = PopularityBaseline()
baseline.fit(train_carts, avg_prices)


def evaluate(model, test_carts, candidates, k_values=[1, 3, 5]):
    results = {k: 0 for k in k_values}
    total = 0

    for cart in test_carts:
        cart_candidates = [p for p in cart if p in candidates]
        if not cart_candidates:
            continue

        for removed in cart_candidates:
            partial_cart = [p for p in cart if p != removed]
            ranking = model.rank(partial_cart, candidates)
            top_products = [p for p, _, _ in ranking]

            for k in k_values:
                if removed in top_products[:k]:
                    results[k] += 1
            total += 1

    return {k: v / total if total > 0 else 0 for k, v in results.items()}, total


print(f"\n=== Ranking Evaluation ===")
print(f"Candidates: {len(CANDIDATES)} products")

nb_results, total = evaluate(nb_model, test_carts, CANDIDATES)
bl_results, _ = evaluate(baseline, test_carts, CANDIDATES)

print(f"\nTotal test cases: {total}")
print(f"\nHit@K:")
print(f"{'K':<5} {'Naive Bayes':<15} {'Popularity':<15}")
print("-" * 35)
for k in [1, 3, 5]:
    print(f"{k:<5} {nb_results[k]:.3f}           {bl_results[k]:.3f}")

print("\n=== Example ===")
example_cart = ["Crazy Schnitzel", "French Fries"]
print(f"Cart: {example_cart}")
print("Top 5:")
ranking = nb_model.rank(example_cart, CANDIDATES)
for product, score, prob in ranking[:5]:
    price = avg_prices.get(product, 0)
    print(f"  {product}: score={score:.4f}, price={price:.2f}")
