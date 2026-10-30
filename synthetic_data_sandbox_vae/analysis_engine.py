"""Analisi dei soli dati sintetici: nessun codice LLM viene eseguito."""
from __future__ import annotations

import json
import os
import re
import pandas as pd

ALLOWED_KINDS = {'correlation', 'histogram', 'average', 'heatmap'}


def _mentioned_columns(question: str, columns: list[str]) -> list[str]:
    query = question.casefold()
    result = []
    for col in columns:
        if re.search(r'(?<!\w)' + re.escape(col.casefold()) + r'(?!\w)', query):
            result.append(col)
    return result


def _heuristic_plan(question: str, df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include='number').columns.tolist()
    mentioned = _mentioned_columns(question, df.columns.tolist())
    nums = [c for c in mentioned if c in numeric]
    lower = question.casefold()
    if 'correl' in lower or 'relazione' in lower:
        preferred = nums or [c for c in ['reddito', 'risparmio'] if c in numeric]
        return {'kind': 'correlation', 'columns': (preferred + [c for c in numeric if c not in preferred])[:2]}
    if 'heatmap' in lower or 'matrice' in lower:
        return {'kind': 'heatmap', 'columns': numeric[:12]}
    if 'media' in lower or 'medio' in lower or 'average' in lower:
        return {'kind': 'average', 'columns': (nums or numeric)[:1]}
    return {'kind': 'histogram', 'columns': (nums or numeric)[:1]}


def _llm_plan(question: str, df: pd.DataFrame) -> dict | None:
    if not os.getenv('OPENAI_API_KEY'):
        return None
    try:
        from openai import OpenAI
        schema = [{'name': col, 'type': str(df[col].dtype)} for col in df.columns]
        client = OpenAI(timeout=15.0)
        response = client.chat.completions.create(
            model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
            messages=[
                {'role': 'system', 'content': (
                    'Sei un interprete di domande su dati sintetici. Non produci codice. '
                    'Rispondi ESCLUSIVAMENTE con JSON {"kind":"...", "columns":["..."]}. '
                    'kind deve essere correlation, histogram, average oppure heatmap. '
                    'Scegli solo nomi di colonne nello schema. correlation vuole due colonne numeriche; '
                    'histogram e average una colonna numerica; heatmap colonne numeriche. '
                    'Schema: ' + json.dumps(schema, ensure_ascii=False))},
                {'role': 'user', 'content': question[:700]}
            ], response_format={'type': 'json_object'})
        return json.loads(response.choices[0].message.content)
    except Exception:
        # L'app funziona anche offline o senza crediti API
        return None


def analyze_synthetic(question: str, df: pd.DataFrame) -> dict:
    if df.empty:
        raise ValueError('Genera prima i dati sintetici.')
    numeric = df.select_dtypes(include='number').columns.tolist()
    plan = _llm_plan(question, df) or _heuristic_plan(question, df)
    kind = plan.get('kind')
    if kind not in ALLOWED_KINDS:
        plan = _heuristic_plan(question, df)
        kind = plan['kind']
    columns = plan.get('columns', [])
    if not isinstance(columns, list):
        columns = []
    columns = [c for c in columns if isinstance(c, str) and c in numeric]
    fallback = _heuristic_plan(question, df)['columns']
    if kind == 'correlation':
        if len(columns) < 2:
            columns = (columns + [c for c in fallback + numeric if c not in columns])[:2]
        if len(columns) < 2:
            raise ValueError('Servono almeno due colonne numeriche per una correlazione.')
        a, b = columns[:2]
        corr = df[a].corr(df[b])
        value = round(float(corr), 4) if pd.notna(corr) else None
        return {'kind': kind, 'columns': [a, b], 'value': value,
                'answer': f'Correlazione Pearson fra {a} e {b}: {value}.'}
    if kind in {'histogram', 'average'}:
        columns = (columns or [c for c in fallback if c in numeric] or numeric)[:1]
        if not columns:
            raise ValueError('Non ci sono colonne numeriche da analizzare.')
        col = columns[0]
        if kind == 'average':
            value = round(float(df[col].mean()), 2)
            return {'kind': kind, 'columns': [col], 'value': value,
                    'answer': f'La media sintetica di {col} è {value}.'}
        return {'kind': kind, 'columns': [col],
                'answer': f'Ecco la distribuzione sintetica di {col}.'}
    columns = columns or numeric[:12]
    if len(columns) < 2:
        raise ValueError('Servono almeno due colonne numeriche per una heatmap.')
    return {'kind': 'heatmap', 'columns': columns[:12],
            'answer': 'Matrice di correlazione dei dati sintetici.'}
