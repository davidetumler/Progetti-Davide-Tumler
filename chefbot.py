"""
Progetto M5 - ChefBot AI
========================
1) Computer Vision: Food101 + transfer learning MobileNetV2 (15 classi).
2) Knowledge Retrieval: informazioni recuperate dal file piatti.json.
3) Ricerca semantica: embeddings multilingue e similarita' coseno.

Esempi:
    python chefbot.py train --epochs 8
    python chefbot.py predict --image esempio.jpg
    python chefbot.py search --query "voglio un dolce con miele"
    python chefbot.py info --dish baklava

Per l'interfaccia grafica:
    streamlit run app.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# Cartelle assolute: lo script funziona anche se avviato da un'altra directory.
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "piatti.json"
MODEL_DIR = BASE_DIR / "modelli"
MODEL_PATH = MODEL_DIR / "chefbot_food101.keras"
CLASS_NAMES_PATH = MODEL_DIR / "classi.json"
RESULTS_PATH = MODEL_DIR / "risultati_training.json"
CHARTS_DIR = BASE_DIR / "grafici"

IMAGE_SIZE = 160  # Leggero per un MacBook Air; MobileNetV2 accetta 160x160.
BATCH_SIZE = 24
SEED = 42
TEXT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_database() -> dict:
    """Carica il dizionario JSON delle conoscenze sui piatti."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Database non trovato: {DATA_PATH}")
    with DATA_PATH.open("r", encoding="utf-8") as f:
        db = json.load(f)
    for label, item in db.items():
        required = ("nome", "ingredienti", "calorie_kcal", "porzione_g", "descrizione", "tags")
        missing = [key for key in required if key not in item]
        if missing:
            raise ValueError(f"Piatti JSON: dati mancanti per {label}: {missing}")
    return db


def lookup(label: str, db: dict | None = None) -> dict:
    """Recupera dal database i dati strutturati della classe prevista."""
    if db is None:
        db = load_database()
    if label not in db:
        raise KeyError(f"Piatto '{label}' assente dal database")
    return db[label]


def format_dish(label: str, db: dict) -> str:
    """Restituisce una scheda informativa leggibile nel terminale."""
    item = lookup(label, db)
    return (
        f"\n{'=' * 55}\n"
        f"{item['nome'].upper()} ({label})\n"
        f"{'=' * 55}\n"
        f"Descrizione: {item['descrizione']}\n"
        f"Ingredienti: {', '.join(item['ingredienti'])}\n"
        f"Calorie stimate: {item['calorie_kcal']} kcal "
        f"per porzione indicativa da {item['porzione_g']} g\n"
        f"Caratteristiche: {', '.join(item['tags'])}\n"
        f"Nota: calorie indicative, non valori misurati per la foto.\n"
    )


# ============================================================
# PARTE 1: COMPUTER VISION - FOOD101 + MOBILENETV2
# ============================================================

def subset_for_class(ds, original_id: int, new_id: int, limit: int, skip: int = 0):
    """Filtra una classe Food101, prende un sottoinsieme e rimappa la label.

    TFDS legge immagini come byte JPEG, non le decodifica fino a questa map:
    cosi' il filtro sulle etichette evita decodifiche non necessarie.
    """
    import tensorflow as tf

    def is_class(image_bytes, label):
        return tf.equal(label, tf.cast(original_id, label.dtype))

    def decode_and_resize(image_bytes, label):
        image = tf.image.decode_image(
            image_bytes, channels=3, expand_animations=False
        )
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        # Pixel ancora in [0, 255], normalizzazione DENTRO il modello.
        return image, tf.cast(new_id, tf.int32)

    filtered = ds.filter(is_class).skip(skip).take(limit)
    return filtered.map(decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)


def concat_class_subsets(raw_ds, original_ids, take_per_class, skip_per_class=0,
                         mix_classes=False):
    """Preleva lo stesso numero per classe, senza caricare tutto in memoria."""
    import tensorflow as tf

    parts = []
    for new_id, original_id in enumerate(original_ids):
        parts.append(subset_for_class(
            raw_ds, original_id, new_id, take_per_class, skip=skip_per_class
        ))
    if mix_classes:
        # Alterna le classi senza dover caricare migliaia di immagini nel buffer.
        return tf.data.Dataset.sample_from_datasets(parts, seed=SEED)
    ds = parts[0]
    for part in parts[1:]:
        ds = ds.concatenate(part)
    return ds


def prepare_datasets(db: dict, train_per_class: int, val_per_class: int,
                     test_per_class: int):
    """Scarica/prepara Food101 e crea train, validazione e test DISGIUNTI.

    Il dataset TFDS originale ha train e validation. La 'validation'
    viene suddivisa senza sovrapposizione in val e test del progetto.
    """
    import tensorflow as tf
    import tensorflow_datasets as tfds

    if val_per_class + test_per_class > 250:
        raise ValueError("Food101 ha 250 immagini per classe nella split validation.")
    if train_per_class > 750:
        raise ValueError("Food101 ha 750 immagini per classe nella split train.")

    print("Caricamento Food101 da TensorFlow Datasets...")
    print("ATTENZIONE: al primo avvio TFDS prepara l'intero Food101 (~4,65 GiB di download).")
    # Disattiviamo shuffle_files per lasciare fissa la divisione val/test.
    (raw_train, raw_val), info = tfds.load(
        "food101",
        split=["train", "validation"],
        as_supervised=True,
        with_info=True,
        shuffle_files=False,
        decoders={"image": tfds.decode.SkipDecoding()},
    )

    class_names = list(db.keys())  # Ordine identico alle uscite del classificatore.
    original_ids = []
    for name in class_names:
        try:
            original_ids.append(info.features["label"].str2int(name))
        except ValueError as exc:
            raise ValueError(f"La classe '{name}' non esiste in Food101") from exc

    train_ds = concat_class_subsets(
        raw_train, original_ids, train_per_class, mix_classes=True
    )
    val_ds = concat_class_subsets(raw_val, original_ids, val_per_class)
    test_ds = concat_class_subsets(
        raw_val, original_ids, test_per_class, skip_per_class=val_per_class
    )

    train_ds = (
        train_ds.shuffle(buffer_size=min(300, train_per_class * len(class_names)), seed=SEED)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, test_ds, class_names


def build_model(num_classes: int):
    """Costruisce MobileNetV2 con base ImageNet congelata e testa personalizzata."""
    import tensorflow as tf

    tf.keras.utils.set_random_seed(SEED)

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    backbone.trainable = False  # Transfer learning: congelamento dei pesi.

    image_input = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="immagine_rgb")
    x = tf.keras.layers.RandomFlip("horizontal", name="augmentation_flip")(image_input)
    x = tf.keras.layers.RandomRotation(0.05, name="augmentation_rotazione")(x)
    # Equivalente a mobilenet_v2.preprocess_input: [0,255] -> [-1,1].
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="normalizzazione")(x)
    x = backbone(x, training=False)
    x = tf.keras.layers.Dropout(0.30)(x)
    output = tf.keras.layers.Dense(num_classes, activation="softmax", name="classi_food101")(x)
    model = tf.keras.Model(inputs=image_input, outputs=output, name="ChefBot_MobileNetV2")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def save_training_chart(history):
    """Salva grafici accuracy/loss utili per la relazione del progetto."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history.history["loss"]) + 1)
    axs[0].plot(epochs, history.history["accuracy"], label="Training")
    axs[0].plot(epochs, history.history["val_accuracy"], label="Validazione")
    axs[0].set(xlabel="Epoca", ylabel="Accuracy", title="Accuratezza")
    axs[0].legend()
    axs[1].plot(epochs, history.history["loss"], label="Training")
    axs[1].plot(epochs, history.history["val_loss"], label="Validazione")
    axs[1].set(xlabel="Epoca", ylabel="Loss", title="Errore")
    axs[1].legend()
    fig.tight_layout()
    path = CHARTS_DIR / "training.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"Grafico salvato: {path}")


def train(args):
    """Addestra il classificatore e ne valuta le prestazioni sui dati di test."""
    import tensorflow as tf

    db = load_database()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Classi selezionate ({len(db)}): {', '.join(db.keys())}")
    train_ds, val_ds, test_ds, class_names = prepare_datasets(
        db, args.train_per_class, args.val_per_class, args.test_per_class
    )
    model = build_model(len(class_names))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5
        ),
    ]
    print("\nAvvio addestramento...")
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs,
        callbacks=callbacks, verbose=1
    )

    print("\nValutazione finale su immagini MAI usate per l'addestramento o early stopping:")
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
    print(f"Test accuracy: {test_accuracy:.2%} | Test loss: {test_loss:.4f}")

    model.save(MODEL_PATH)
    with CLASS_NAMES_PATH.open("w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    metrics = {
        "classi": class_names,
        "numero_classi": len(class_names),
        "immagini_train_per_classe": args.train_per_class,
        "immagini_val_per_classe": args.val_per_class,
        "immagini_test_per_classe": args.test_per_class,
        "epoche_effettuate": len(history.history["loss"]),
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "history": {key: [float(value) for value in values]
                    for key, values in history.history.items()},
    }
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    save_training_chart(history)
    print(f"Modello salvato: {MODEL_PATH}")
    print(f"Etichette salvate: {CLASS_NAMES_PATH}")
    print(f"Metriche salvate: {RESULTS_PATH}")


def load_vision_model():
    """Carica modello e classi; serve aver eseguito prima 'train'."""
    import tensorflow as tf

    if not MODEL_PATH.is_file() or not CLASS_NAMES_PATH.is_file():
        raise FileNotFoundError(
            "Modello non addestrato. Esegui prima: python chefbot.py train --epochs 8"
        )
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with CLASS_NAMES_PATH.open("r", encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names


def predict_pil(image: Image.Image, model, class_names: list[str], top_k: int = 3):
    """Predice piatti da una foto Pillow; output = [(classe, probabilita'), ...]."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    # Stessi pixel raw [0,255] del training: il modello normalizza da se'.
    image_array = np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)
    probabilities = model.predict(image_array, verbose=0)[0]
    top_indices = np.argsort(probabilities)[::-1][:min(top_k, len(class_names))]
    return [(class_names[int(i)], float(probabilities[i])) for i in top_indices]


def predict_file(image_path: str):
    db = load_database()
    model, class_names = load_vision_model()
    with Image.open(image_path) as image:
        predictions = predict_pil(image, model, class_names)
    print("\nTOP 3 PREDIZIONI:")
    for rank, (label, prob) in enumerate(predictions, start=1):
        print(f" {rank}. {db[label]['nome']}: {prob:.1%}")
    best_label = predictions[0][0]
    print(format_dish(best_label, db))
    if predictions[0][1] < 0.50:
        print("Attenzione: predizione incerta, verificare il risultato visivamente.")
    print("Nota: il classificatore conosce SOLO le 15 classi del progetto.")


# ============================================================
# PARTE 2 e 3: KNOWLEDGE RETRIEVAL + EMBEDDINGS SEMANTICI
# ============================================================

def document_for_embedding(label: str, db: dict) -> str:
    """Testo ricco che il modello NLP trasforma in un vettore numerico."""
    item = db[label]
    return (
        f"{item['nome']}. {item['descrizione']} "
        f"Ingredienti: {', '.join(item['ingredienti'])}. "
        f"Tipo di piatto e caratteristiche: {', '.join(item['tags'])}."
    )


def load_text_model():
    """Primo uso: scarica il modello multilingue di Sentence Transformers."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(TEXT_MODEL_NAME)


def build_semantic_index(text_model, db: dict):
    """Calcola una volta gli embeddings NORMALIZZATI per tutti i 15 piatti."""
    class_names = list(db.keys())
    documents = [document_for_embedding(name, db) for name in class_names]
    embeddings = text_model.encode(
        documents, normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=False
    )
    return class_names, np.asarray(embeddings, dtype=np.float32)


def semantic_search(query: str, text_model, class_names: list[str],
                    embeddings: np.ndarray, top_k: int = 3):
    """Similarita' coseno: per vettori unitari coincide col prodotto scalare."""
    if not query.strip():
        raise ValueError("La ricerca non puo' essere vuota.")
    query_embedding = text_model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=False
    )[0]
    scores = embeddings @ np.asarray(query_embedding, dtype=np.float32)
    best_indices = np.argsort(scores)[::-1][:min(top_k, len(class_names))]
    return [(class_names[int(i)], float(scores[i])) for i in best_indices]


def search_cli(query: str, top_k: int):
    db = load_database()
    print("Caricamento embeddings multilingue...")
    text_model = load_text_model()
    class_names, vectors = build_semantic_index(text_model, db)
    results = semantic_search(query, text_model, class_names, vectors, top_k)
    print(f"\nRicerca: {query!r}")
    for rank, (label, score) in enumerate(results, start=1):
        print(f" {rank}. {db[label]['nome']}: similarita' coseno = {score:.3f}")
    if results:
        print(format_dish(results[0][0], db))
    print("Nota: la similarita' coseno NON e' una probabilita' di correttezza.")


def main():
    parser = argparse.ArgumentParser(
        description="ChefBot AI: visione, database di ricette e ricerca semantica"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Addestra MobileNetV2 su 15 classi Food101")
    p_train.add_argument("--epochs", type=int, default=8)
    p_train.add_argument("--train-per-class", type=int, default=150)
    p_train.add_argument("--val-per-class", type=int, default=35)
    p_train.add_argument("--test-per-class", type=int, default=35)

    p_predict = sub.add_parser("predict", help="Riconosci il piatto in una foto")
    p_predict.add_argument("--image", required=True, help="Percorso della foto JPG/PNG")

    p_search = sub.add_parser("search", help="Cerca un piatto usando una frase")
    p_search.add_argument("--query", required=True, help="Es. 'un dolce con miele'")
    p_search.add_argument("--top-k", type=int, default=3)

    p_info = sub.add_parser("info", help="Mostra la scheda di un piatto")
    p_info.add_argument("--dish", required=True, help="Etichetta, es. baklava")

    args = parser.parse_args()
    try:
        if args.command == "train":
            if min(args.epochs, args.train_per_class, args.val_per_class, args.test_per_class) < 1:
                parser.error("Epoche e numeri di immagini devono essere >= 1.")
            train(args)
        elif args.command == "predict":
            predict_file(args.image)
        elif args.command == "search":
            search_cli(args.query, args.top_k)
        elif args.command == "info":
            print(format_dish(args.dish, load_database()))
    except (FileNotFoundError, KeyError, ValueError) as exc:
        parser.exit(status=1, message=f"ERRORE: {exc}\n")


if __name__ == "__main__":
    main()
