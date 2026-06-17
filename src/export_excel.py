import pandas as pd
import os
import json

os.makedirs("results", exist_ok=True)

with open(
    "results/history.json",
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)

df = pd.DataFrame(data)

excel_path = "results/history.xlsx"

df.to_excel(
    excel_path,
    index=False
)

print("\nExcel report created:")
print(excel_path)

print("\nRows exported:")
print(len(df))