"""Confronta una domanda con i pezzi di testo estratti dai PDF."""
from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .documents import Passage

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@dataclass(frozen=True)
class SearchResult:
    passage: Passage
    score: float


class SearchEngine:
    def __init__(self, passages, mode="tfidf", model=None):
        if not passages:
            raise ValueError("Non c'è testo su cui cercare.")
        if mode not in ("tfidf", "embeddings"):
            raise ValueError("Tipo di ricerca non valido.")

        self.passages = passages
        self.mode = mode
        self.model = model
        testi = [pezzo.text for pezzo in passages]

        if mode == "tfidf":
            # TF-IDF dà un peso alle parole presenti nei documenti.
            self.vectorizer = TfidfVectorizer(strip_accents="unicode", ngram_range=(1, 2), sublinear_tf=True)
            self.matrix = self.vectorizer.fit_transform(testi)
        else:
            if model is None:
                raise ValueError("Manca il modello per gli embeddings.")
            # Il modello trasforma ogni pezzo di testo in una lista di numeri.
            self.matrix = np.asarray(model.encode(testi, normalize_embeddings=True))

    def search(self, query, top_k=3, document=None):
        if not query.strip():
            return []

        if self.mode == "tfidf":
            domanda = self.vectorizer.transform([query])
            soglia = 0.035
        else:
            domanda = np.asarray(self.model.encode([query], normalize_embeddings=True))
            soglia = 0.30

        # La similarità coseno ci aiuta a mettere i risultati in ordine.
        punteggi = cosine_similarity(domanda, self.matrix).ravel()
        ordine = np.argsort(punteggi)[::-1]
        trovati = []

        for indice in ordine:
            pezzo = self.passages[int(indice)]
            punteggio = float(punteggi[indice])
            if punteggio < soglia:
                continue
            if document is not None and pezzo.document != document:
                continue
            trovati.append(SearchResult(pezzo, punteggio))
            if len(trovati) == top_k:
                break
        return trovati
