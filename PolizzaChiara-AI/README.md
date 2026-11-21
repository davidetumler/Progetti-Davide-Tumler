# PolizzaChiara AI

Questo è il progetto che ho preparato per il Capstone del Master Python e Intelligenza Artificiale di EPICODE.

L'idea è semplice: quando una polizza è molto lunga, trovare una clausola può richiedere tempo. Con questa applicazione posso caricare un PDF, scrivere una domanda e vedere le parti del documento più vicine a quello che sto cercando, con il numero della pagina.

Ho aggiunto anche un piccolo confronto tra due polizze. Non è un programma che decide se un sinistro è coperto e non sostituisce la lettura delle condizioni.

## Cosa fa

- Legge uno o due PDF con testo selezionabile.
- Divide il testo in parti più piccole, mantenendo la pagina.
- Cerca le parti che assomigliano di più alla domanda.
- Mostra il testo originale trovato.
- Mette vicini due estratti di polizze diverse per confrontare uno stesso argomento.
- Ha due PDF inventati per fare le prove.

## Programmi e librerie usate

- **Python**: il linguaggio con cui è scritto il progetto.
- **Streamlit**: crea la pagina web che vedo nel browser.
- **PyMuPDF**: legge i PDF.
- **Scikit-learn**: serve per la ricerca TF-IDF e la similarità coseno.
- **NumPy**: aiuta nei calcoli.
- **Sentence Transformers (facoltativo)**: permette di provare anche una ricerca più avanzata con un modello già addestrato.

## Come farlo partire sul Mac

1. Scarica lo ZIP e apri la cartella `PolizzaChiara_AI`.
2. Apri il **Terminale** del Mac.
3. Scrivi `cd ` (con uno spazio), trascina la cartella dentro il Terminale e premi Invio.
4. Copia i comandi seguenti, **uno per volta**:

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Se sul Mac hai più versioni di Python, è consigliata la 3.11. Per chiudere l'applicazione premi `Control + C` nel Terminale. Per farla ripartire, riapri il Terminale nella cartella, attiva l'ambiente e usa l'ultimo comando.

Dopo l'avvio si apre una pagina nel browser. Se non compare, prova `http://localhost:8501`.

## Come provarlo

1. Lascia attivata la casella **Usa le due polizze di esempio**.
2. Seleziona **Base (TF-IDF)**.
3. Apri la scheda **Fai una domanda**.
4. Scegli, per esempio, **Qual è la franchigia per i danni da acqua?** e premi **Cerca**.
5. Guarda i risultati: troverai testo, nome del PDF e numero di pagina.
6. Apri **Confronta due polizze**, scegli **Danni da acqua** e premi **Confronta**.

## Ricerca avanzata (facoltativa)

Se vuoi provare la seconda modalità, prima installa questa libreria aggiuntiva:

```bash
python -m pip install -r requirements-ai.txt
```

Poi riavvia il programma e seleziona **Avanzata (embeddings)**. La prima volta deve scaricare il modello da Internet. Il modello è già addestrato: **non l'ho addestrato da zero**. Se non funziona, il programma torna alla ricerca base.

## Test

Dalla cartella principale, con l'ambiente attivo:

```bash
python -m pip install pytest
python -m pytest -q
python -m evaluation.valuta_ricerca
```

I test controllano alcune funzioni e verificano le risposte su otto domande preparate usando i PDF di esempio. Non provano che il programma capisca tutte le polizze del mondo.

## Come sono organizzati i file

- `app.py`: la pagina del programma.
- `core/documents.py`: legge il PDF e divide il testo.
- `core/search.py`: cerca i passaggi più vicini alla domanda.
- `core/comparison.py`: confronta lo stesso tema in due PDF.
- `dati_demo/`: due polizze inventate.
- `tests/`: test automatici.
- `evaluation/`: domande usate per provare la ricerca.
- `PITCH_3_MINUTI.md`: discorso e indicazioni per il video.
- `GUIDA_COLLOQUIO.md`: cose da ripassare per spiegare il progetto.

## Limiti che conosco

Per ora il programma legge solo PDF che contengono testo, quindi non legge bene i documenti scansionati come immagini. Inoltre restituisce estratti e non una risposta legale. Le parole di una domanda possono influire sui risultati e il confronto non dice quale polizza sia migliore.

Per migliorarlo aggiungerei il riconoscimento del testo nelle scansioni (OCR), altri test e una ricerca più precisa sulle clausole.
