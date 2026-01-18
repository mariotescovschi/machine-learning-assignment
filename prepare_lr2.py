import json
import pandas as pd
from datetime import datetime

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

with open("./data/receipts.json", "r") as f:
    receipts = json.load(f)

data = []

for r in receipts:
    items = r["items"]
    products = [i["retail_product_name"] for i in items]
    dt = datetime.fromisoformat(r["date"])

    row = {
        "id_bon": r["id_bon"],
        "day_of_week": dt.isoweekday(),
        "is_weekend": 1 if dt.isoweekday() >= 6 else 0,
        "hour": dt.hour,
        "cart_size": len(items),
        "distinct_products": len(set(products)),
        "total_value": sum(i["SalePriceWithVAT"] for i in items),
    }

    # Label for each sauce
    for sauce in SAUCES:
        row[f"label_{sauce.replace(' ', '_').lower()}"] = 1 if sauce in products else 0

    # Product counts (exclude all sauces)
    for name in products:
        if name in SAUCES:
            continue
        col = f"prod_{name.replace(' ', '_').lower()}"
        row[col] = row.get(col, 0) + 1

    data.append(row)

df = pd.DataFrame(data).fillna(0)
df.to_csv("data/dataset_lr2.csv", index=False)
print(f"{len(df)} receipts saved to data/dataset_lr2.csv")

# Sauce popularity stats
print("\nSauce distribution:")
for sauce in SAUCES:
    col = f"label_{sauce.replace(' ', '_').lower()}"
    count = int(df[col].sum())
    pct = count / len(df) * 100
    print(f"  {sauce}: {count} ({pct:.1f}%)")
