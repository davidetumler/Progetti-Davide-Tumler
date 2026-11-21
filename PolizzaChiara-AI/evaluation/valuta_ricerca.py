"""Prova alcune domande e controlla se troviamo il testo giusto."""
import json
from pathlib import Path
from core.documents import extract_passages
from core.search import SearchEngine

cartella = Path(__file__).resolve().parents[1]
passaggi = []
for pdf in sorted((cartella / "dati_demo").glob("*.pdf")):
    passaggi.extend(extract_passages(pdf.read_bytes(), pdf.name))

ricerca = SearchEngine(passaggi)
domande = json.loads((cartella / "evaluation" / "quesiti.json").read_text(encoding="utf-8"))
corrette = 0

for numero, prova in enumerate(domande, start=1):
    risultati = ricerca.search(prova["question"], document=prova["document"])
    trovato = any(prova["expected"].lower() in r.passage.text.lower() for r in risultati)
    if trovato:
        corrette += 1
    print(f"{'OK' if trovato else 'NO'} - {numero}. {prova['question']}")

print(f"\nRisultati trovati: {corrette} su {len(domande)}")
print("Questa prova controlla la ricerca sui PDF di esempio, non la correttezza assicurativa delle risposte.")
