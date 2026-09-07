# Progetto 4

## Riconoscimento gesti della mano - Sasso, Carta, Forbice

In questo progetto ho realizzato una rete neurale convoluzionale (CNN) in Python con TensorFlow/Keras per riconoscere tre gesti della mano: sasso, carta e forbice.

Per l'addestramento ho utilizzato il dataset Rock-Paper-Scissors, composto da immagini sintetiche divise nelle tre classi.

Le immagini vengono ridimensionate a 150x150 pixel e normalizzate. Durante il training viene applicata anche la Data Augmentation, utilizzando rotazioni, spostamenti, shear e ribaltamento orizzontale.

La CNN è composta da più livelli Conv2D e MaxPooling2D, seguiti da un livello Flatten, un Dense da 512 neuroni e un livello finale Softmax con 3 classi.

Il modello è stato addestrato con Adam e categorical_crossentropy. Il training è stato impostato per un massimo di 20 epoche, con una callback che interrompe l'addestramento al superamento del 98% di accuracy.

Il risultato finale sul dataset di test è stato:

**Test Accuracy: 95.97%**

Dopo il training ho provato il modello anche con la webcam integrata del Mac.

Il gesto sasso è stato riconosciuto molto bene, mentre carta ha avuto qualche difficoltà in più. Il gesto forbice, invece, veniva spesso riconosciuto come sasso.

Questa differenza è dovuta al fatto che il modello è stato addestrato su immagini sintetiche molto pulite, mentre con la webcam entrano in gioco luce, ombre, sfondo e posizione reale della mano.

## File

- `PROGETTO_FINALE_RPS_MAC.py` - contiene il training, la CNN e la valutazione finale.
- `TEST_WEBCAM_RPS_MAC.py` - permette di provare il modello con la webcam del Mac.

Il file del modello `.keras` non è presente nel repository perché troppo grande. Viene comunque creato automaticamente dopo il training.
