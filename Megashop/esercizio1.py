import glob
import pandas as pd
import dask.dataframe as dd

# ---------------------------------------
# PANDAS
# ---------------------------------------

print("ESERCIZIO PANDAS")

files = glob.glob(
    "dataset/transazioni_json/*.jsonl"
)

totale = 0

for file in files:

    df = pd.read_json(
        file,
        lines=True
    )

    somma = df["amount"].sum()

    totale = totale + somma

    print(file)
    print("Totale file:", somma)

print("\nTotale generale:")
print(totale)

# ---------------------------------------
# DASK
# ---------------------------------------

print("\nESERCIZIO DASK")

ddf = dd.read_json(
    "dataset/transazioni_json/*.jsonl",
    lines=True
)

risultato = (
    ddf.groupby("region_id")["amount"]
    .mean()
    .compute()
)

print("\nMedia amount per region_id")
print(risultato)