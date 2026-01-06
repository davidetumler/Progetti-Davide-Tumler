#Progetto: Gestione Biblioteca Digitale (Python)
# In una biblioteca digitale si vuole realizzare un piccolo sistema software per gestire libri, utenti e prestiti.Il programma deve sfruttare variabili, tipi di dati, strutture di controllo e soprattutto la programmazione orientata agli oggetti (OOP).

#Parte 1 – Variabili e tipi di dati
# Variabili di esempio

titolo = "Naruto"
copie = 5
prezzo = 14.99
disponibile = True

print("Titolo:", titolo)
print("Copie disponibili:", copie)
print("Prezzo medio:", prezzo)
print("Disponibile:", disponibile)

# Parte 2 – Strutture dati
# Lista di libri

libri = [
    "Naruto",
    "One Piece",
    "Dragonball",
    "Bleach",
    "Attack on Titan"
]

# Dizionario: copie disponibili
copie_libri = {
    "Naruto": 5,
    "One Piece": 3,
    "Dragonball": 4,
    "Bleach": 6,
    "Attack on Titan": 2
}

# Set di utenti registrati
utenti_registrati = {"Davide", "Federico", "Mario", "Nicole"}

print(libri)
print(copie_libri)
print(utenti_registrati)

# Parte 3 – Classi e OOP

class Libro:
    def __init__(self, titolo, autore, anno, copie_disponibili):
        self.titolo = titolo
        self.autore = autore
        self.anno = anno
        self.copie_disponibili = copie_disponibili

    def info(self):
        return f"{self.titolo} - {self.autore} ({self.anno}) | Copie: {self.copie_disponibili}"


class Utente:
    def __init__(self, nome, eta, id_utente):
        self.nome = nome
        self.eta = eta
        self.id_utente = id_utente

    def scheda(self):
        print(f"Utente: {self.nome}, Età: {self.eta}, ID: {self.id_utente}")


class Prestito:
    def __init__(self, utente, libro, giorni):
        self.utente = utente
        self.libro = libro
        self.giorni = giorni

    def dettagli(self):
        print(f"{self.utente.nome} ha preso '{self.libro.titolo}' per {self.giorni} giorni")

# Parte 4 – Funzionalità

def presta_libro(utente, libro, giorni):
    if libro.copie_disponibili > 0:
        libro.copie_disponibili -= 1
        prestito = Prestito(utente, libro, giorni)
        return prestito
    else:
        print(f"Errore: '{libro.titolo}' non è disponibile")
        return None

#Simulazione dei prestiti

# Creazione libri
libro1 = Libro("Naruto", "Kishimoto", 1999, 2)
libro2 = Libro("One Piece", "Oda", 1997, 1)
libro3 = Libro("Dragonball", "Toriyama", 1984, 3)

# Creazione utenti
utente1 = Utente("Davide", 20, 1)
utente2 = Utente("Federico", 22, 2)
utente3 = Utente("Mario", 19, 3)

# Prestiti
prestiti = []

prestiti.append(presta_libro(utente1, libro1, 7))
prestiti.append(presta_libro(utente2, libro2, 10))
prestiti.append(presta_libro(utente3, libro3, 5))

# Stampa copie aggiornate
print("\nCopie disponibili:")
for libro in [libro1, libro2, libro3]:
    print(libro.info())

# Stampa dettagli prestiti
print("\nDettagli prestiti:")
for p in prestiti:
    if p:
        p.dettagli()
        
# Fine del programma