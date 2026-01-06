# Progetto – Analisi Vendite Negozio di Elettronica

# Parte 1 – Dataset vendite.csv

# Parte 2 – Importazione con Pandas

import pandas as pd

df = pd.read_csv("/Users/davidetumler/Desktop/vendite.csv", sep=";",encoding="latin1")
df.columns = ["Data", "Negozio", "Prodotto", "Quantità", "Prezzo_unitario"]

print(df.head())
print(df.shape)
print(df.info())

# Parte 3 – Elaborazioni con Pandas

df["Incasso"] = df["Quantità"] * df["Prezzo_unitario"]

print("Incasso totale:", df["Incasso"].sum())

print("\nIncasso medio per negozio:")
print(df.groupby("Negozio")["Incasso"].mean())

print("\n3 prodotti più venduti:")
print(df.groupby("Prodotto")["Quantità"].sum().sort_values(ascending=False).head(3))

print("\nIncasso medio per negozio e prodotto:")
print(df.groupby(["Negozio", "Prodotto"])["Incasso"].mean())

#Parte 4 – NumPy

import numpy as np

q = df["Quantità"].to_numpy()

print("Media:", np.mean(q))
print("Min:", np.min(q))
print("Max:", np.max(q))
print("Deviazione standard:", np.std(q))

percentuale = np.sum(q > np.mean(q)) / len(q) * 100
print("Vendite sopra la media:", percentuale, "%")

array_2d = df[["Quantità", "Prezzo_unitario"]].to_numpy()
incassi_numpy = array_2d[:, 0] * array_2d[:, 1]

print("Incassi corretti:", np.allclose(incassi_numpy, df["Incasso"]))

# Parte 5 – Grafici

import matplotlib.pyplot as plt

df.groupby("Negozio")["Incasso"].sum().plot(kind="bar")
plt.title("Incasso totale per negozio")
plt.ylabel("Euro")
plt.show()

df.groupby("Prodotto")["Incasso"].sum().plot(kind="pie", autopct="%1.1f%%")
plt.title("Incasso per prodotto")
plt.ylabel("")
plt.show()

df["Data"] = pd.to_datetime(df["Data"])
df.groupby("Data")["Incasso"].sum().plot(marker="o")
plt.title("Andamento giornaliero incassi")
plt.ylabel("Euro")
plt.show()

# Parte 6 – Analisi Avanzata (Categorie)
def categoria(prodotto):
    if prodotto == "iPhone":
        return "Elettronica"
    else:
        return "Elettrodomestici"

df["Categoria"] = df["Prodotto"].apply(categoria)

print(df.groupby("Categoria").agg(
    Incasso_totale=("Incasso", "sum"),
    Quantità_media=("Quantità", "mean")
))

df.to_csv("vendite_analizzate.csv", index=False)

#fine programma