import json
import pandas as pd
from datetime import datetime

with open("./data/receipts.json", "r") as f:
    receipts = json.load(f)

data = []

for r in receipts:
    items = r["items"]
    products = [i["retail_product_name"] for i in items]

    if "Crazy Schnitzel" not in products:
        continue

    dt = datetime.fromisoformat(r["date"])

    row = {
        "id_bon": r["id_bon"],
        "label": 1 if "Crazy Sauce" in products else 0,
        "day_of_week": dt.isoweekday(),
        "is_weekend": 1 if dt.isoweekday() >= 6 else 0,
        "hour": dt.hour,
        "cart_size": len(items),
        "distinct_products": len(set(products)),
        "total_value": sum(i["SalePriceWithVAT"] for i in items),
    }

    for name in products:
        if name == "Crazy Sauce":
            continue
        col = f"prod_{name.replace(' ', '_').lower()}"
        row[col] = row.get(col, 0) + 1

    data.append(row)

df = pd.DataFrame(data).fillna(0)
df.to_csv("data/dataset_lr1.csv", index=False)
print(f"{len(df)} samples saved")
