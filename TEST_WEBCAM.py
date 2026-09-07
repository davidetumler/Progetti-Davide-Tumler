import os
import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = (150, 150)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "rps_cnn_model.keras")

CLASS_NAMES = {
    0: "CARTA",
    1: "SASSO",
    2: "FORBICE"
}

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        "Modello rps_cnn_model.keras non trovato nella cartella del progetto."
    )

print("Caricamento modello...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Modello caricato correttamente.")

# Webcam integrata del Mac tramite AVFoundation
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    raise RuntimeError(
        "Impossibile aprire la webcam. Controlla i permessi della fotocamera."
    )

print("")
print("Webcam attiva.")
print("Premi SPAZIO per classificare il gesto.")
print("Premi Q per uscire.")
print("")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Errore nella lettura della webcam.")
        break

    cv2.putText(
        frame,
        "SPAZIO = predici | Q = esci",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Rock Paper Scissors - Webcam", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == 32:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, IMG_SIZE)
        image = resized.astype("float32") / 255.0
        image = np.expand_dims(image, axis=0)

        prediction = model.predict(image, verbose=0)[0]

        predicted_index = int(np.argmax(prediction))
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(prediction[predicted_index]) * 100.0

        print(
            "Predizione: "
            + predicted_class
            + " - Confidenza: "
            + f"{confidence:.2f}%"
        )

cap.release()
cv2.destroyAllWindows()
print("Webcam chiusa.")
