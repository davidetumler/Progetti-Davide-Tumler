from pathlib import Path
from core.documents import Passage, extract_passages
from core.search import SearchEngine
from core.comparison import compare_topic

DEMO = Path(__file__).resolve().parents[1] / 'dati_demo'


def make_engine():
    passages = []
    for pdf in sorted(DEMO.glob('*.pdf')):
        passages.extend(extract_passages(pdf.read_bytes(), pdf.name))
    return SearchEngine(passages)


def test_search_finds_furto():
    engine = make_engine()
    hits = engine.search('La polizza prevede il furto?', document='polizza_casa_plus.pdf')
    assert hits and 'furto' in hits[0].passage.text.lower()


def test_search_filters_document():
    engine = make_engine()
    hits = engine.search('franchigia danni da acqua', document='polizza_casa_base.pdf')
    assert hits and all(hit.passage.document == 'polizza_casa_base.pdf' for hit in hits)


def test_comparison_returns_both_documents():
    engine = make_engine()
    left, right = compare_topic(engine, 'polizza_casa_base.pdf', 'polizza_casa_plus.pdf', 'Danni da acqua')
    assert left is not None and right is not None
    assert '350' in left.passage.text and '100' in right.passage.text


def test_empty_query():
    engine = SearchEngine([Passage('demo.pdf', 1, 'Acqua e incendio', 0)])
    assert engine.search('   ') == []
