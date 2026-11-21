"""Legge un PDF e divide il testo in pezzi da usare nella ricerca."""
from dataclasses import dataclass
import re
import pymupdf

MAX_FILE_MB = 20


@dataclass(frozen=True)
class Passage:
    document: str
    page: int
    text: str
    position: int


def normalize_text(testo):
    """Sistema gli spazi senza cambiare le parole."""
    testo = testo.replace("\xa0", " ").replace("\x00", "")
    testo = re.sub(r"[ \t]+", " ", testo)
    testo = re.sub(r"\n{3,}", "\n\n", testo)
    return testo.strip()


def dividi_testo(testo, lunghezza=850, sovrapposizione=100):
    """Se il testo è troppo lungo, lo divide cercando di non tagliare le parole."""
    if len(testo) <= lunghezza:
        return [testo] if testo else []

    pezzi = []
    inizio = 0
    while inizio < len(testo):
        fine = min(inizio + lunghezza, len(testo))
        if fine < len(testo):
            spazio = testo.rfind(" ", inizio + lunghezza // 2, fine)
            if spazio != -1:
                fine = spazio
        pezzo = testo[inizio:fine].strip()
        if pezzo:
            pezzi.append(pezzo)
        if fine == len(testo):
            break
        inizio = max(fine - sovrapposizione, inizio + 1)
    return pezzi


def extract_passages(pdf_bytes, filename):
    """Ritorna il testo con il nome del PDF e il numero di pagina."""
    if not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError(f"'{filename}' non sembra un PDF valido.")
    if len(pdf_bytes) > MAX_FILE_MB * 1024 * 1024:
        raise ValueError(f"'{filename}' supera il limite di {MAX_FILE_MB} MB.")

    passaggi = []
    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as pdf:
            if pdf.needs_pass:
                raise ValueError(f"'{filename}' è protetto da password.")

            for numero_pagina, pagina in enumerate(pdf, start=1):
                blocco_attuale = []

                # Quando un blocco è completo lo aggiungiamo alla lista.
                def salva_blocco():
                    testo = normalize_text("\n".join(blocco_attuale))
                    for pezzo in dividi_testo(testo):
                        passaggi.append(Passage(filename, numero_pagina, pezzo, len(passaggi)))
                    blocco_attuale.clear()

                for blocco in pagina.get_text("blocks", sort=True):
                    if len(blocco) < 7 or blocco[6] != 0:
                        continue
                    testo = normalize_text(blocco[4])
                    if not testo or testo.startswith("PolizzaChiara AI | EPICODE Capstone Project"):
                        continue

                    # I titoli numerati sono un buon punto per dividere le clausole.
                    nuovo_titolo = bool(re.match(r"^\d{1,3}[.)]\s+\S", testo))
                    nuovo_titolo = nuovo_titolo or (len(testo) < 90 and testo.isupper())
                    if blocco_attuale and (nuovo_titolo or sum(len(r) for r in blocco_attuale) + len(testo) > 700):
                        salva_blocco()
                    blocco_attuale.append(testo)

                if blocco_attuale:
                    salva_blocco()
    except (pymupdf.FileDataError, pymupdf.EmptyFileError) as errore:
        raise ValueError(f"Impossibile leggere '{filename}': {errore}") from errore

    return passaggi
