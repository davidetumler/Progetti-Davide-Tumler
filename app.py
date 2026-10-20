"""Interfaccia web di ChefBot M5 (avvio: streamlit run app.py)."""

import streamlit as st
from PIL import Image

from chefbot import (
    MODEL_PATH,
    CLASS_NAMES_PATH,
    load_database,
    load_vision_model,
    predict_pil,
    load_text_model,
    build_semantic_index,
    semantic_search,
)

st.set_page_config(page_title="ChefBot AI", page_icon="🍝", layout="centered")


@st.cache_resource(show_spinner="Caricamento modello per il riconoscimento immagini...")
def vision_engine():
    return load_vision_model()


@st.cache_resource(show_spinner="Caricamento modello per la ricerca semantica...")
def text_engine():
    model = load_text_model()
    db = load_database()
    names, vectors = build_semantic_index(model, db)
    return model, names, vectors


def show_dish(label: str, db: dict):
    dish = db[label]
    st.subheader(dish["nome"])
    st.write(dish["descrizione"])
    st.write("**Ingredienti principali:** " + ", ".join(dish["ingredienti"]))
    a, b = st.columns(2)
    a.metric("Calorie stimate", f"{dish['calorie_kcal']} kcal")
    b.metric("Porzione indicativa", f"{dish['porzione_g']} g")
    st.caption("Valori nutrizionali indicativi del piatto, non ricavati dalla singola foto.")


st.title("🍽️ ChefBot AI")
st.write("**Progetto M5:** riconoscimento immagini, conoscenza alimentare e ricerca semantica.")

try:
    db = load_database()
except (OSError, ValueError) as exc:
    st.error(f"Problema nel database JSON: {exc}")
    st.stop()

vision_tab, search_tab, database_tab = st.tabs([
    "📷 Riconosci un piatto", "💬 Ricerca per umore", "📚 Database"
])

with vision_tab:
    st.write("Carica la fotografia di uno dei 15 piatti studiati dal classificatore.")
    uploaded = st.file_uploader("Scegli una foto JPG/PNG", type=["jpg", "jpeg", "png"])
    if not MODEL_PATH.is_file() or not CLASS_NAMES_PATH.is_file():
        st.info("Prima addestra il classificatore dal terminale con: python chefbot.py train --epochs 8")
    if uploaded is not None:
        try:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Immagine caricata", use_container_width=True)
            if MODEL_PATH.is_file() and CLASS_NAMES_PATH.is_file():
                with st.spinner("Analisi dell'immagine..."):
                    model, class_names = vision_engine()
                    predictions = predict_pil(image, model, class_names, top_k=3)
                best_label, best_prob = predictions[0]
                st.success(f"Piatto riconosciuto: {db[best_label]['nome']} ({best_prob:.1%})")
                if best_prob < 0.50:
                    st.warning("Predizione poco sicura: controlla anche le altre ipotesi.")
                st.write("**Prime tre ipotesi:**")
                for label, prob in predictions:
                    st.write(f"• {db[label]['nome']}: {prob:.1%}")
                show_dish(best_label, db)
                st.caption("Il modello distingue solo le classi del progetto: non identifica piatti sconosciuti.")
        except Exception as exc:
            st.error(f"Errore durante il riconoscimento: {exc}")

with search_tab:
    st.write("Prova: 'vorrei qualcosa di fresco e leggero' oppure 'dolce con miele'.")
    query = st.text_input("Come ti senti? Che cosa vorresti mangiare?")
    if st.button("Trova il piatto", type="primary"):
        if not query.strip():
            st.warning("Inserisci una frase prima di cercare.")
        else:
            try:
                model, names, embeddings = text_engine()
                matches = semantic_search(query, model, names, embeddings, top_k=3)
                st.write("**Piatti consigliati per similarità semantica:**")
                for i, (label, score) in enumerate(matches, start=1):
                    st.write(f"{i}. **{db[label]['nome']}** — coseno: {score:.3f}")
                show_dish(matches[0][0], db)
                st.caption("Il coseno è una misura di somiglianza tra embeddings, non una probabilità.")
            except Exception as exc:
                st.error(f"Problema nella ricerca semantica: {exc}")

with database_tab:
    selected = st.selectbox(
        "Scegli un piatto", options=list(db.keys()), format_func=lambda key: db[key]["nome"]
    )
    show_dish(selected, db)
