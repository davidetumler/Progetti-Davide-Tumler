"""Cerca lo stesso argomento in due documenti diversi."""
from .search import SearchEngine

# Per il confronto uso queste domande preparate in anticipo.
TOPICS = {
    "Incendio": "garanzia incendio danni provocati dal fuoco massimale",
    "Danni da acqua": "danni da acqua rottura accidentale tubazioni ricerca guasto franchigia",
    "Furto": "furto rapina abitazione indennizzo scoperto",
    "Responsabilità civile": "responsabilità civile verso terzi massimale danni involontari",
    "Assistenza": "assistenza emergenza idraulico elettricista invio tecnico",
}


def compare_topic(engine: SearchEngine, document_a, document_b, topic):
    if topic not in TOPICS:
        raise ValueError("Non conosco questo argomento.")

    domanda = TOPICS[topic]
    primo = engine.search(domanda, top_k=1, document=document_a)
    secondo = engine.search(domanda, top_k=1, document=document_b)
    return (primo[0] if primo else None, secondo[0] if secondo else None)
