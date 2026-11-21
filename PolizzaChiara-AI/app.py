"""Piccola app per cercare informazioni dentro le polizze in PDF."""
from pathlib import Path
import streamlit as st

from core.documents import extract_passages
from core.search import SearchEngine, MODEL_NAME
from core.comparison import TOPICS, compare_topic

CARTELLA = Path(__file__).parent

st.set_page_config(page_title="PolizzaChiara AI", page_icon="📄", layout="wide")
st.title("📄 PolizzaChiara AI")
st.write("Un modo semplice per cercare informazioni dentro una polizza assicurativa.")
st.info("È un progetto per il Master: mostra parti dei documenti, non dà un parere assicurativo.")



# La lettura non viene ripetuta ogni volta che premiamo un pulsante.
@st.cache_data
def leggi_pdf(nome, contenuto):
    return extract_passages(contenuto, nome)



# Il modello viene caricato una sola volta quando scegliamo la ricerca avanzata.
@st.cache_resource
def carica_modello():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def mostra_risultato(risultato):
    testo = risultato.passage
    st.markdown(f"**{testo.document} - pagina {testo.page}**")
    st.write(testo.text)
    st.caption(f"Somiglianza con la domanda: {risultato.score:.2f}. Non è una percentuale di affidabilità.")


# Prima scegliamo i documenti e il tipo di ricerca.
with st.sidebar:
    st.header("Impostazioni")
    usa_demo = st.checkbox("Usa le due polizze di esempio", value=True)
    scelta = st.radio("Tipo di ricerca", ["Base (TF-IDF)", "Avanzata (embeddings)"])
    st.caption("La ricerca base è più leggera. Quella avanzata usa un modello già addestrato e richiede un'installazione aggiuntiva.")
    st.caption("Per le prove usa documenti inventati o pubblici, senza dati dei clienti.")

file_pdf = []
if usa_demo:
    for percorso in sorted((CARTELLA / "dati_demo").glob("*.pdf")):
        file_pdf.append((percorso.name, percorso.read_bytes()))
    st.success(f"Ho caricato {len(file_pdf)} polizze di esempio.")
else:
    caricati = st.file_uploader("Carica uno o due PDF (massimo 20 MB per file)", type="pdf", accept_multiple_files=True)
    if len(caricati) > 2:
        st.warning("Per ora il programma usa solo i primi due PDF.")
    for file in caricati[:2]:
        file_pdf.append((file.name, file.getvalue()))

if not file_pdf:
    st.info("Per iniziare, carica un PDF oppure attiva i documenti di esempio.")
    st.stop()

# Leggiamo il testo e ricordiamo il numero di pagina.
passaggi = []
for nome, contenuto in file_pdf:
    try:
        passaggi.extend(leggi_pdf(nome, contenuto))
    except ValueError as errore:
        st.error(str(errore))

if not passaggi:
    st.warning("Non riesco a leggere il testo. I PDF scansionati richiedono l'OCR, che non è presente in questa versione.")
    st.stop()

# Prepariamo la ricerca. Se il modello avanzato non funziona, usiamo quello base.
if scelta.startswith("Avanzata"):
    try:
        with st.spinner("Sto caricando il modello per la ricerca avanzata..."):
            motore = SearchEngine(passaggi, mode="embeddings", model=carica_modello())
    except (ImportError, OSError, RuntimeError, ValueError) as errore:
        st.warning("Non riesco ad avviare la ricerca avanzata. Continuo con quella base.")
        st.caption(str(errore))
        motore = SearchEngine(passaggi)
else:
    motore = SearchEngine(passaggi)

nomi = sorted(set(p.document for p in passaggi))
st.caption(f"Documenti: {len(nomi)} | Parti di testo: {len(passaggi)} | Ricerca: {motore.mode}")

scheda1, scheda2, scheda3 = st.tabs(["Fai una domanda", "Confronta due polizze", "Come funziona"])

with scheda1:
    st.subheader("Cerca nella polizza")
    esempio = st.selectbox("Oppure prova una di queste domande", [
        "Scrivo io la domanda",
        "Qual è la franchigia per i danni da acqua?",
        "Qual è il massimale della responsabilità civile?",
        "La polizza copre il furto?",
    ])
    testo_iniziale = "" if esempio == "Scrivo io la domanda" else esempio
    domanda = st.text_input("La tua domanda", value=testo_iniziale)
    documento = st.selectbox("Dove vuoi cercare?", ["Tutti i documenti"] + nomi)

    if st.button("Cerca", type="primary"):
        if not domanda.strip():
            st.warning("Prima scrivi una domanda.")
        else:
            filtro = None if documento == "Tutti i documenti" else documento
            risultati = motore.search(domanda, top_k=3, document=filtro)
            if not risultati:
                st.warning("Non ho trovato passaggi abbastanza simili. Questo non significa che la garanzia non ci sia.")
            else:
                for numero, risultato in enumerate(risultati, start=1):
                    with st.container(border=True):
                        st.write(f"Risultato {numero}")
                        mostra_risultato(risultato)
                st.info("Per sicurezza bisogna sempre leggere la clausola completa nel PDF originale.")

with scheda2:
    st.subheader("Confronto tra due documenti")
    if len(nomi) < 2:
        st.info("Per il confronto servono due PDF.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            primo = st.selectbox("Prima polizza", nomi, index=0)
        with col2:
            secondo = st.selectbox("Seconda polizza", nomi, index=1)
        argomento = st.selectbox("Che cosa vuoi confrontare?", list(TOPICS))
        if st.button("Confronta", type="primary"):
            if primo == secondo:
                st.warning("Scegli due polizze diverse.")
            else:
                sinistra, destra = compare_topic(motore, primo, secondo, argomento)
                a, b = st.columns(2)
                with a:
                    st.markdown(f"**{primo}**")
                    mostra_risultato(sinistra) if sinistra else st.warning("Non ho trovato un passaggio.")
                with b:
                    st.markdown(f"**{secondo}**")
                    mostra_risultato(destra) if destra else st.warning("Non ho trovato un passaggio.")
                st.info("Il programma mette vicini due estratti: non sceglie quale polizza è migliore.")

with scheda3:
    st.write("""
    **In breve:**
    1. Carico un PDF.
    2. Il programma legge il testo e lo divide in parti più piccole.
    3. Confronta la domanda con queste parti.
    4. Mostra quelle più simili e la pagina da cui arrivano.

    La prima modalità usa TF-IDF. La seconda usa embeddings creati da un modello linguistico già addestrato.
    Non c'è un chatbot che inventa risposte: qui vengono mostrati i testi originali.
    """)
