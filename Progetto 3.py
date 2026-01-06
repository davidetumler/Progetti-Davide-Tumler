 # PARTE 1 – VARIABILI E TIPI DI DATI


nome = "Davide Tumler"
eta = 27
saldo = 3000.00
vip = True

destinazioni = ["Budapest", "Firenze", "Tokyo", "New York", "Londra"]

prezzi_medi = {
    "Budapest": 400,
    "Firenze": 600,
    "Tokyo": 1500,
    "New York": 2000,
    "Londra": 1200
}


# PARTE 2 – OOP

class Cliente:
    def __init__(self, nome, eta, vip):
        self.nome = nome
        self.eta = eta
        self.vip = vip

    def stampa_info(self):
        print("Cliente:", self.nome, "- Età:", self.eta, "- VIP:", self.vip)


class Viaggio:
    def __init__(self, destinazione, prezzo, durata):
        self.destinazione = destinazione
        self.prezzo = prezzo
        self.durata = durata


class Prenotazione:
    def __init__(self, cliente, viaggio):
        self.cliente = cliente
        self.viaggio = viaggio

    def importo_finale(self):
        if self.cliente.vip:
            return self.viaggio.prezzo * 0.9
        return self.viaggio.prezzo

    def dettagli(self):
        self.cliente.stampa_info()
        print("Destinazione:", self.viaggio.destinazione)
        print("Durata:", self.viaggio.durata, "giorni")
        print("Prezzo finale:", self.importo_finale(), "€")



# PARTE 3 – NUMPY


import numpy as np

prezzi = np.random.uniform(200, 2000, 100)

media = np.mean(prezzi)
minimo = np.min(prezzi)
massimo = np.max(prezzi)
dev_std = np.std(prezzi)
percentuale = np.sum(prezzi > media) / len(prezzi) * 100

print("\nSTATISTICHE PREZZI")
print("Media:", media)
print("Minimo:", minimo)
print("Massimo:", massimo)
print("Deviazione standard:", dev_std)
print("Percentuale sopra la media:", percentuale, "%")


# PARTE 4 – PANDAS


import pandas as pd

np.random.seed(1)

dati = {
    "Cliente": np.random.choice(["Mario", "Luisa", "Anna", "Paolo", "Giulia"], 100),
    "Destinazione": np.random.choice(destinazioni, 100),
    "Prezzo": np.random.uniform(200, 2000, 100),
    "Giorno_Partenza": np.random.randint(1, 31, 100),
    "Durata": np.random.randint(3, 15, 100)
}

df = pd.DataFrame(dati)
df["Incasso"] = df["Prezzo"]

incasso_totale = df["Incasso"].sum()
incasso_medio = df.groupby("Destinazione")["Incasso"].mean()
top_3 = df["Destinazione"].value_counts().head(3)

print("\nINCASSO TOTALE:", incasso_totale)
print("\nINCASSO MEDIO PER DESTINAZIONE:\n", incasso_medio)
print("\nTOP 3 DESTINAZIONI:\n", top_3)


# PARTE 5 – MATPLOTLIB


import matplotlib.pyplot as plt

df.groupby("Destinazione")["Incasso"].sum().plot(kind="bar")
plt.title("Incasso per Destinazione")
plt.show()

df.groupby("Giorno_Partenza")["Incasso"].sum().plot()
plt.title("Andamento Giornaliero Incassi")
plt.show()

df["Destinazione"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("Percentuale Vendite per Destinazione")
plt.ylabel("")
plt.show()


# PARTE 6 – ANALISI AVANZATA


categorie = {
    "Roma": "Europa",
    "Parigi": "Europa",
    "Tokyo": "Asia",
    "New York": "America",
    "Il Cairo": "Africa"
}

df["Categoria"] = df["Destinazione"].map(categorie)

incasso_categoria = df.groupby("Categoria")["Incasso"].sum()
durata_media_categoria = df.groupby("Categoria")["Durata"].mean()

print("\nINCASSO PER CATEGORIA:\n", incasso_categoria)
print("\nDURATA MEDIA PER CATEGORIA:\n", durata_media_categoria)

df.to_csv("prenotazioni_analizzate.csv", index=False)


# PARTE 7 – ESTENSIONI


# Funzione: N clienti con più prenotazioni
def top_clienti(df, n):
    return df["Cliente"].value_counts().head(n)

print("\nTOP 3 CLIENTI:\n", top_clienti(df, 3))

# Grafico combinato
fig, ax1 = plt.subplots()

incasso_categoria.plot(kind="bar", ax=ax1, color="skyblue")
ax1.set_ylabel("Incasso Medio")

ax2 = ax1.twinx()
durata_media_categoria.plot(kind="line", ax=ax2, color="red", marker="o")
ax2.set_ylabel("Durata Media")

plt.title("Incasso e Durata Media per Categoria")
plt.show()

#fine programma