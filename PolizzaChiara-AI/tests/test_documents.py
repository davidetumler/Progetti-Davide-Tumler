from pathlib import Path
import pytest
from core.documents import extract_passages, normalize_text

DEMO = Path(__file__).resolve().parents[1] / 'dati_demo'


def test_invalid_pdf_rejected():
    with pytest.raises(ValueError, match='PDF valido'):
        extract_passages(b'not a pdf', 'fake.pdf')


def test_text_normalization():
    assert normalize_text('  Ciao   mondo\n\n\nfine  ') == 'Ciao mondo\n\nfine'


def test_pdf_extraction_has_page_citation():
    file = DEMO / 'polizza_casa_plus.pdf'
    passages = extract_passages(file.read_bytes(), file.name)
    assert passages
    assert all(p.page >= 1 for p in passages)
    assert any('25.000' in p.text and 'Franchigia' in p.text for p in passages)


def test_pdf_extraction_keeps_furto_clause_together():
    file = DEMO / 'polizza_casa_base.pdf'
    passages = extract_passages(file.read_bytes(), file.name)
    assert any('Garanzia furto' in p.text and 'NON è presente' in p.text for p in passages)
