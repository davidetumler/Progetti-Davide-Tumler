# Progetto 2

#Un centro di analisi mediche deve informatizzare parte della gestione dei pazienti, dei medici e dei referti di laboratorio. Si richiede la progettazione e la realizzazione di un programma in Python che permetta di gestire i dati in maniera strutturata, utilizzando programmazione a oggetti (OOP) e la libreria NumPy per l’elaborazione numerica dei dati clinici.

# Paziente 1
nome1 = "Davide"
cognome1 = "Rossi"
codice_fiscale1 = "RSSDVD97A01H501Z"
eta1 = 28
peso1 = 95.5
analisi1 = ["glicemia", "colesterolo", "trigliceridi"]

# Paziente 2
nome2 = "Laura"
cognome2 = "Bianca"
codice_fiscale2 = "BNCLRA81B41F205X"
eta2 = 44
peso2 = 67.4
analisi2 = ["glicemia", "pressione"]

# Paziente 3
nome3 = "Andrea"
cognome3 = "Verdone"
codice_fiscale3 = "VRDNDR73C15L219Y"
eta3 = 52
peso3 = 75.0
analisi3 = ["glicemia", "pressione"]

#  PARTE 2 – Classi e OOP

import numpy as np

class Paziente:
    def __init__(self, nome, cognome, codice_fiscale, eta, peso, analisi_effettuate, risultati_analisi):
        self.nome = nome
        self.cognome = cognome
        self.codice_fiscale = codice_fiscale
        self.eta = eta
        self.peso = peso
        self.analisi_effettuate = analisi_effettuate
        self.risultati_analisi = np.array(risultati_analisi)

    def scheda_personale(self):
        return f"""
Paziente: {self.nome} {self.cognome}
Codice Fiscale: {self.codice_fiscale}
Età: {self.eta} anni
Peso: {self.peso} kg
Analisi effettuate: {', '.join(self.analisi_effettuate)}
"""

    def statistiche_analisi(self):
        return {
            "media": np.mean(self.risultati_analisi),
            "minimo": np.min(self.risultati_analisi),
            "massimo": np.max(self.risultati_analisi),
            "deviazione_standard": np.std(self.risultati_analisi)
        }


class Medico:
    def __init__(self, nome, cognome, specializzazione):
        self.nome = nome
        self.cognome = cognome
        self.specializzazione = specializzazione

    def visita_paziente(self, paziente):
        print(f"Il medico {self.nome} {self.cognome} visita il paziente {paziente.nome} {paziente.cognome}")


class Analisi:
    def __init__(self, tipo, risultato):
        self.tipo = tipo
        self.risultato = risultato

    def valuta(self):
        if self.tipo == "glicemia":
            return "Normale" if 70 <= self.risultato <= 110 else "Fuori norma"
        elif self.tipo == "colesterolo":
            return "Normale" if self.risultato < 200 else "Alto"
        else:
            return "Valore non valutabile"

# PARTE 3 – Uso di NumPy

valori_glicemia = np.array([90, 85, 110, 95, 100, 88, 92, 105, 98, 87])

print("Media:", np.mean(valori_glicemia))
print("Massimo:", np.max(valori_glicemia))
print("Minimo:", np.min(valori_glicemia))
print("Deviazione standard:", np.std(valori_glicemia))


#  PARTE 5 – Applicazione Completa (MAIN)

# Medici
medico1 = Medico("Chiara", "Neri", "Endocrinologia")
medico2 = Medico("Lucia", "Marzio", "Cardiologia")
medico3 = Medico("Andrea", "Contini", "Medicina Generale")

# Pazienti
paziente1 = Paziente(
    "Davide", "Rossi", "RSSDVD97A01H501Z", 28, 95.5,
    ["glicemia", "colesterolo", "trigliceridi"],
    [95, 180, 140]
)

paziente2 = Paziente(
    "Laura", "Bianca", "BNCLRA81B41F205X", 44, 67.4,
    ["glicemia", "colesterolo", "pressione"],
    [88, 210, 120]
)

paziente3 = Paziente(
    "Andrea", "Verdone", "VRDNDR73C15L219Y", 52, 75.0,
    ["glicemia", "pressione", "colesterolo"],
    [110, 130, 190]
)

if __name__ == "__main__":
    # Stampa schede personali
    print(paziente1.scheda_personale())
    print(paziente2.scheda_personale())
    print(paziente3.scheda_personale())

    # Visite mediche
    medico1.visita_paziente(paziente1)
    medico2.visita_paziente(paziente2)
    medico3.visita_paziente(paziente3)

    # fine programma