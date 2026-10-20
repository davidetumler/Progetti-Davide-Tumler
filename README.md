# Progetto M5 – ChefBot AI (Python)

Progetto finale Food-Tech con **15 classi Food101**:
`baklava`, `caesar_salad`, `cheesecake`, `chocolate_cake`, `churros`,
`french_fries`, `greek_salad`, `hamburger`, `ice_cream`, `lasagna`, `pizza`,
`ramen`, `spaghetti_carbonara`, `sushi`, `tiramisu`.

## 1. Aprire in VS Code sul Mac

Aprire in VS Code la cartella `ChefBot_M5`. Nel terminale integrato:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Se hai già Python 3.11 configurato come `python3`, puoi utilizzare `python3` al posto di `python3.11`.

**Importante:** TensorFlow e Sentence Transformers usano due modelli distinti: il primo per le foto, il secondo per i testi. Al primo uso scaricano anche i rispettivi pesi.

## 2. Prima prova SENZA addestramento

Il database JSON e la ricerca semantica si possono testare subito:

```bash
python chefbot.py info --dish baklava
python chefbot.py search --query "voglio un dolce con miele"
python chefbot.py search --query "cerco qualcosa di fresco e leggero"
```

`search` richiede l'installazione di `sentence-transformers` e il download iniziale del suo modello multilingue.

## 3. Addestrare il classificatore Food101

```bash
python chefbot.py train --epochs 8
```

- Dataset da **TensorFlow Datasets**: `food101`, suddiviso nelle split originali `train` e `validation`.
- **ATTENZIONE ALLA MEMORIA/DISCO:** anche se addestriamo solo 15 classi, TFDS al primo avvio scarica/prepara l'intero dataset. Dimensione di download riportata nella scheda ufficiale: circa **4,65 GiB**. Occorre spazio aggiuntivo per l'estrazione/preparazione del dataset. Non chiudere il terminale durante il download.
- Le immagini vengono processate con `tf.data` in streaming, senza caricare il dataset completo nella RAM.
- A ogni classe assegniamo un numero locale tra 0 e 14 (invece delle etichette originali 0–100).
- Parametri predefiniti: **150 immagini/classi per train**, **35 per validazione**, **35 per test**. Val e test vengono separati SENZA sovrapposizioni dalla split TFDS `validation` (che contiene 250 immagini per classe).
- Per un avvio più rapido sul Mac puoi fare una prova con:

```bash
python chefbot.py train --epochs 3 --train-per-class 60 --val-per-class 20 --test-per-class 20
```

La riduzione delle immagini riguarda il training, **non la preparazione iniziale** del dataset.

### Modello

1. Input 160 × 160 RGB.
2. Data augmentation (flip + piccole rotazioni).
3. Normalizzazione dei pixel da [0, 255] a [-1, 1] (coerente con MobileNetV2).
4. `MobileNetV2(weights='imagenet', include_top=False, pooling='avg')` con pesi **congelati**.
5. `Dropout(0.30)` e livello `Dense(15, softmax)` allenabile.
6. Loss: cross-entropia per etichette intere (`SparseCategoricalCrossentropy`).
7. `EarlyStopping` e `ReduceLROnPlateau` per ottimizzare l'apprendimento.
8. Valutazione indipendente su un sottoinsieme **test** escluso da train e validazione.

File creati dopo il training:

- `modelli/chefbot_food101.keras` — modello addestrato.
- `modelli/classi.json` — etichette nell'ordine delle 15 uscite.
- `modelli/risultati_training.json` — epoche, loss, accuracy e test accuracy misurata.
- `grafici/training.png` — grafici delle curve di apprendimento.

L'accuratezza **non è nota prima dell'addestramento**: i risultati dipendono dall'hardware, dalla quantità di dati e dal training effettuato.

## 4. Predire una foto

Dopo il training, posiziona una foto nella cartella del progetto, per esempio `mia_pizza.jpg`:

```bash
python chefbot.py predict --image "mia_pizza.jpg"
```

ChefBot restituisce le **prime tre classi** con le rispettive probabilità del modello e recupera la scheda informativa del piatto più probabile dal JSON.

**Limite:** è un classificatore closed-set: se la foto raffigura un cibo diverso dai 15 conosciuti, il modello fornirà comunque una delle 15 classi.

## 5. Interfaccia grafica (consigliata per la presentazione)

```bash
streamlit run app.py
```

Si aprirà una pagina web locale con tre sezioni:

- **Riconosci un piatto:** caricamento immagine, classificazione top 3 e scheda nutrizionale.
- **Ricerca per umore:** frase in italiano, suggerimenti tramite embeddings e similarità coseno.
- **Database:** esplorazione delle 15 schede.

La ricerca semantica funziona **anche prima** di addestrare MobileNetV2.

## 6. Come funziona la ricerca semantica

Per ogni piatto, lo script unisce nome, descrizione, ingredienti e tag in un testo. Il modello
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` lo trasforma in un vettore (embedding).
Trasforma in vettore anche la frase dell'utente e confronta la sua direzione con quella dei piatti:

`similarita_coseno(A, B) = (A · B) / (||A|| × ||B||)`

Nel codice gli embeddings sono normalizzati: la formula diventa un semplice prodotto scalare.
Il punteggio misura la somiglianza semantica, **non** una probabilità. I piatti disponibili sono soltanto quelli presenti nel JSON.

## 7. Precisazioni per la relazione

- I valori nutrizionali sono **stime illustrative per porzione**, non calcoli effettuati dalle immagini né informazioni verificate su ricette univoche. Variano in base a quantità, ingredienti e preparazione.
- Food101 include 101 classi e 101.000 immagini; qui studiamo un subset bilanciato di 15 classi.
- Lo split del progetto è: training dalle immagini ufficiali train; validazione e test da porzioni **disgiunte** della split ufficiale validation.
- Le classi del JSON devono coincidere esattamente con i nomi delle classi TFDS.
- Per migliorare l'accuratezza: aumentare `--train-per-class`, le epoche (con EarlyStopping), cambiare risoluzione, valutare fine tuning delle ultime parti della rete.
- La normalizzazione [0,255] → [-1,1] viene svolta **dentro il modello** e resta corretta anche usando Streamlit o foto esterne.

## 8. Risoluzione dei problemi

**Errore 'No module named ...'**: attiva `.venv` e installa `pip install -r requirements.txt`.

**Non trova il modello**: esegui `python chefbot.py train --epochs 8` prima di caricare foto.

**Download Food101 lento/spazio insufficiente**: è normale che il primo download sia consistente; una volta completata la cache, gli avvii successivi la riutilizzano. Se il Mac non ha abbastanza spazio, puoi eseguire la fase di addestramento su Google Colab e riportare i file `modelli/` sul Mac.

**Foto non riconosciuta**: prova un'immagine di uno dei piatti previsti; non è addestrato su tutte le 101 classi.

**Avvertenze TensorFlow sul Mac**: alcuni messaggi relativi alla CPU/GPU sono informativi; in assenza di GPU il training funziona su CPU, ma può essere più lento.

## Fonti tecniche

- Food101 TFDS: https://www.tensorflow.org/datasets/catalog/food101
- MobileNetV2: https://keras.io/api/applications/mobilenet/mobilenet_models/
- Embeddings multilingue: https://www.sbert.net/examples/sentence_transformer/training/multilingual/README.html

## 9. Verifica rapida del codice (senza scaricare modelli)

```bash
python -m unittest test_chefbot.py -v
```

Questo controlla lettura JSON, formattazione delle schede, input della funzione di classificazione e matematica della ricerca coseno con modelli fittizi. **Non** dimostra l'accuratezza di MobileNetV2 o la qualità reale degli embeddings: quelle richiedono l'esecuzione dei modelli veri.
