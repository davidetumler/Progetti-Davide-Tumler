# ============================================================
# PROGETTO FINALE - RICONOSCIMENTO GESTI DELLA MANO
# CNN con Data Augmentation per Sasso, Carta, Forbice
# Versione pensata per Mac
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


# ============================================================
# 1. IMPOSTAZIONI PRINCIPALI
# ============================================================

# Dimensione richiesta dal progetto
IMG_SIZE = (150, 150)

# Numero di immagini processate insieme.
# Il caricamento a batch evita di saturare la RAM.
BATCH_SIZE = 32

# Numero massimo di epoche richiesto dal progetto
EPOCHS = 20

# Cartella in cui si trova questo file Python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Le due cartelle devono essere estratte e posizionate
# nella stessa cartella di questo script.
TRAIN_DIR = os.path.join(BASE_DIR, "rps")
TEST_DIR = os.path.join(BASE_DIR, "rps-test-set")

MODEL_PATH = os.path.join(BASE_DIR, "rps_cnn_model.keras")


# ============================================================
# 2. CONTROLLO CARTELLE DEL DATASET
# ============================================================

def controlla_dataset():
    """
    Controlla che le cartelle del training e del test esistano
    e che contengano le tre sottocartelle richieste.
    """
    cartelle_richieste = ["rock", "paper", "scissors"]

    for directory, nome in [(TRAIN_DIR, "TRAINING"), (TEST_DIR, "TEST")]:
        if not os.path.isdir(directory):
            raise FileNotFoundError(
                f"\nERRORE: cartella {nome} non trovata:\n{directory}\n\n"
                "Metti le cartelle 'rps' e 'rps-test-set' nella stessa "
                "cartella del file Python."
            )

        for classe in cartelle_richieste:
            percorso = os.path.join(directory, classe)
            if not os.path.isdir(percorso):
                raise FileNotFoundError(
                    f"Manca la sottocartella '{classe}' dentro:\n{directory}"
                )

    print("Dataset trovato correttamente.")


# ============================================================
# 3. DATA PIPELINE E DATA AUGMENTATION
# ============================================================

def crea_generatori():
    """
    Crea i generatori per training e test.

    Sul training vengono applicate:
    - normalizzazione
    - rotazione
    - spostamento orizzontale
    - shear
    - flip orizzontale
    - fill_mode nearest

    Sul test applichiamo solamente la normalizzazione.
    """

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=40,
        width_shift_range=0.2,
        shear_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0
    )

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    print("\nClassi trovate:", train_generator.class_indices)

    return train_generator, test_generator


# ============================================================
# 4. CREAZIONE DELLA CNN
# ============================================================

def crea_modello():
    """
    CNN sequenziale con 4 blocchi convoluzionali.
    I filtri aumentano progressivamente:
    32 -> 64 -> 128 -> 128

    Dopo le convoluzioni:
    Flatten -> Dense 512 -> Dropout -> Softmax 3 classi
    """

    model = Sequential([
        Input(shape=(150, 150, 3)),

        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),

        Flatten(),

        Dense(512, activation="relu"),

        # Aiuta a ridurre l'overfitting
        Dropout(0.5),

        # Tre neuroni: rock, paper, scissors
        Dense(3, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# 5. CALLBACK PERSONALIZZATA: STOP AL 98%
# ============================================================

class StopAccuracy98(tf.keras.callbacks.Callback):
    """
    Interrompe il training quando l'accuracy di training
    raggiunge o supera il 98%.
    """

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        accuracy = logs.get("accuracy")

        if accuracy is not None and accuracy >= 0.98:
            print(
                "\nAccuracy >= 98%. "
                "Interrompo anticipatamente l'addestramento."
            )
            self.model.stop_training = True


# ============================================================
# 6. GRAFICI DEL TRAINING
# ============================================================

def mostra_grafici(history):
    """
    Mostra andamento di accuracy e loss durante le epoche.
    """

    epoche = range(1, len(history.history["accuracy"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epoche, history.history["accuracy"], label="Training accuracy")

    if "val_accuracy" in history.history:
        plt.plot(epoche, history.history["val_accuracy"], label="Test accuracy")

    plt.xlabel("Epoca")
    plt.ylabel("Accuracy")
    plt.title("Andamento Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(epoche, history.history["loss"], label="Training loss")

    if "val_loss" in history.history:
        plt.plot(epoche, history.history["val_loss"], label="Test loss")

    plt.xlabel("Epoca")
    plt.ylabel("Loss")
    plt.title("Andamento Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# 7. PREDIZIONE DI UNA NORMALE IMMAGINE DA FILE
# ============================================================

def predici_immagine(model, image_path, class_indices):
    """
    Predice rock/paper/scissors su una singola immagine.

    L'immagine viene:
    - ridimensionata a 150x150
    - convertita in array
    - normalizzata dividendo per 255
    """

    image = load_img(image_path, target_size=IMG_SIZE)
    image_array = img_to_array(image)
    image_array = image_array / 255.0

    # Il modello si aspetta un batch: aggiungiamo una dimensione
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array, verbose=0)[0]

    # Invertiamo il dizionario:
    # esempio {'paper': 0, 'rock': 1, 'scissors': 2}
    index_to_class = {
        indice: classe for classe, indice in class_indices.items()
    }

    predicted_index = int(np.argmax(prediction))
    predicted_class = index_to_class[predicted_index]
    confidence = float(prediction[predicted_index]) * 100

    print("\n--- RISULTATO IMMAGINE ---")
    print("Predizione:", predicted_class.upper())
    print(f"Confidenza: {confidence:.2f}%")

    return predicted_class, confidence


# ============================================================
# 8. TEST REALE CON WEBCAM SU MAC
# ============================================================

def test_webcam_mac(model, class_indices):
    """
    Apre la webcam del Mac tramite OpenCV.

    COMANDI:
    - SPACE = scatta e classifica la foto
    - Q     = chiude la webcam

    Prima di usare questa funzione:
        pip install opencv-python

    macOS potrebbe chiedere il permesso di usare la fotocamera
    per Terminale o Visual Studio Code.
    """

    try:
        import cv2
    except ImportError:
        print(
            "\nOpenCV non è installato.\n"
            "Installa con:\n"
            "pip install opencv-python"
        )
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print(
            "\nNon riesco ad aprire la webcam.\n"
            "Controlla i permessi della fotocamera in:\n"
            "Impostazioni di Sistema > Privacy e Sicurezza > Fotocamera."
        )
        return

    index_to_class = {
        indice: classe for classe, indice in class_indices.items()
    }

    print("\nWebcam aperta.")
    print("Premi SPAZIO per classificare il gesto.")
    print("Premi Q per uscire.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Errore nella lettura della webcam.")
            break

        cv2.imshow("RPS - Premi SPAZIO per classificare, Q per uscire", frame)

        key = cv2.waitKey(1) & 0xFF

        # Q
        if key == ord("q"):
            break

        # Barra spaziatrice
        if key == 32:
            # OpenCV usa BGR, mentre il modello è stato addestrato in RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Ridimensionamento identico al training
            resized = cv2.resize(rgb_frame, IMG_SIZE)

            # Normalizzazione identica al training
            image_array = resized.astype("float32") / 255.0

            image_array = np.expand_dims(image_array, axis=0)

            prediction = model.predict(image_array, verbose=0)[0]

            predicted_index = int(np.argmax(prediction))
            predicted_class = index_to_class[predicted_index]
            confidence = float(prediction[predicted_index]) * 100

            print(
                f"Predizione webcam: {predicted_class.upper()} "
                f"- confidenza {confidence:.2f}%"
            )

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# 9. PROGRAMMA PRINCIPALE
# ============================================================

def main():
    print("\n==============================================")
    print(" RICONOSCIMENTO SASSO - CARTA - FORBICE")
    print(" CNN CON DATA AUGMENTATION")
    print("==============================================\n")

    # Verifica la presenza del dataset
    controlla_dataset()

    # Generatori di immagini
    train_generator, test_generator = crea_generatori()

    # Creazione del modello
    model = crea_modello()

    print("\nStruttura della CNN:")
    model.summary()

    # Callback richiesta dal progetto
    stop_98 = StopAccuracy98()

    # EarlyStopping aggiuntiva:
    # se la validation loss non migliora per 4 epoche,
    # interrompe il training e ripristina i pesi migliori.
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    )

    print("\nInizio training...\n")

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=test_generator,
        callbacks=[stop_98, early_stopping]
    )

    # ========================================================
    # VALUTAZIONE SUL TEST SET
    # ========================================================

    print("\nValutazione finale sul test set...")

    test_loss, test_accuracy = model.evaluate(
        test_generator,
        verbose=1
    )

    print("\n==============================================")
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print("==============================================")

    # Salvataggio del modello
    model.save(MODEL_PATH)
    print(f"\nModello salvato in:\n{MODEL_PATH}")

    # Grafici
    mostra_grafici(history)

    # ========================================================
    # DOMAIN GAP
    # ========================================================

    print("\n--- ANALISI DEL DOMAIN GAP ---")
    print(
        "Il dataset di training contiene immagini CGI molto pulite, "
        "con sfondo bianco e caratteristiche abbastanza uniformi."
    )
    print(
        "Una foto reale della webcam può invece avere ombre, "
        "illuminazione diversa, sfondo meno uniforme e dettagli "
        "della mano non presenti nelle immagini sintetiche."
    )
    print(
        "Per questo motivo l'accuracy ottenuta sul test CGI può essere "
        "molto alta, mentre sulle fotografie reali il modello può "
        "commettere più errori."
    )
    print(
        "Per ridurre questo Domain Gap è utile usare uno sfondo chiaro, "
        "una buona illuminazione e mantenere la mano ben visibile."
    )

    # ========================================================
    # TEST REALI FACOLTATIVI
    # ========================================================

    print("\n--- TEST REALE FACOLTATIVO ---")
    print(
        "Per provare una foto salvata usa, ad esempio:\n"
        "predici_immagine(model, '/Users/tuonome/Desktop/mia_mano.jpg', "
        "train_generator.class_indices)"
    )

    print(
        "\nPer aprire direttamente la webcam del Mac, "
        "togli il simbolo # dalla riga seguente nel codice:"
    )
    print("# test_webcam_mac(model, train_generator.class_indices)")

    # Per usare direttamente la webcam, togli # dalla riga sotto:
    # test_webcam_mac(model, train_generator.class_indices)


if __name__ == "__main__":
    main()
