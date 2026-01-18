import pandas as pd

df = pd.read_csv("./data/ap_dataset.csv", encoding="utf-8-sig")

df["data_bon"] = pd.to_datetime(df["data_bon"])

# Group by receipt ID
receipts = []

for id_bon, group in df.groupby("id_bon"):
    first = group.iloc[0]

    receipt = {
        "id_bon": int(id_bon),
        "date": first["data_bon"].isoformat(),
        "items": group[["retail_product_name", "SalePriceWithVAT"]].to_dict("records"),
    }
    receipts.append(receipt)

pd.DataFrame(receipts).to_json("./data/receipts.json", orient="records", indent=2)

print(f"Processed {len(receipts)} receipts into receipts.json")
