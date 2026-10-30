"""Preparazione delle colonne tabellari per il modello generativo."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class DataEngine:
    def __init__(self) -> None:
        self.numeric_cols: list[str] = []
        self.category_cols: list[str] = []
        self.numeric_fill: dict[str, float] = {}
        self.category_fill: dict[str, str] = {}
        self.scaler: MinMaxScaler | None = None
        self.encoder: OneHotEncoder | None = None
        self.category_sizes: list[int] = []
        self.feature_count: int = 0

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        if df.empty or len(df) < 20:
            raise ValueError('Il CSV deve contenere almeno 20 righe.')
        if len(df.columns) > 60:
            raise ValueError('Per questa demo sono supportate al massimo 60 colonne.')
        if df.columns.duplicated().any():
            raise ValueError('Il CSV contiene nomi di colonne duplicati.')
        if any(df[c].dropna().map(lambda v: isinstance(v, (dict, list))).any() for c in df):
            raise ValueError('Non sono supportati valori strutturati (liste o dizionari).')

        self.numeric_cols = df.select_dtypes(include='number').columns.tolist()
        self.category_cols = [col for col in df if col not in self.numeric_cols]
        if not self.numeric_cols and not self.category_cols:
            raise ValueError('Nessuna colonna disponibile.')

        clean = self.clean(df, fitting=True)
        features = []
        if self.numeric_cols:
            self.scaler = MinMaxScaler()
            features.append(self.scaler.fit_transform(clean[self.numeric_cols]))
        if self.category_cols:
            try:
                self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            except TypeError:  # compatibilità con versioni meno recenti di sklearn
                self.encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
            categories = self.encoder.fit_transform(clean[self.category_cols])
            self.category_sizes = [len(x) for x in self.encoder.categories_]
            if sum(self.category_sizes) > 400:
                raise ValueError('Troppe categorie: rimuovi identificativi univoci e colonne ad alta cardinalità.')
            features.append(categories)
        array = np.hstack(features).astype(np.float32)
        self.feature_count = array.shape[1]
        return array

    def clean(self, df: pd.DataFrame, fitting: bool = False) -> pd.DataFrame:
        result = df[self.numeric_cols + self.category_cols].copy()
        for col in self.numeric_cols:
            values = pd.to_numeric(result[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
            if fitting:
                median = values.median()
                self.numeric_fill[col] = float(median) if pd.notna(median) else 0.0
            result[col] = values.fillna(self.numeric_fill[col]).astype(float)
        for col in self.category_cols:
            values = result[col].astype('string').fillna('Mancante')
            if fitting:
                mode = values.mode()
                self.category_fill[col] = str(mode.iloc[0]) if len(mode) else 'Mancante'
            result[col] = values.replace('', self.category_fill[col]).astype(str)
        return result

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        clean = self.clean(df)
        parts = []
        if self.numeric_cols:
            parts.append(self.scaler.transform(clean[self.numeric_cols]))
        if self.category_cols:
            parts.append(self.encoder.transform(clean[self.category_cols]))
        return np.hstack(parts).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> pd.DataFrame:
        result: dict[str, object] = {}
        start = 0
        if self.numeric_cols:
            width = len(self.numeric_cols)
            numeric = self.scaler.inverse_transform(np.clip(values[:, :width], 0, 1))
            for index, column in enumerate(self.numeric_cols):
                result[column] = np.round(numeric[:, index], 2)
            start = width
        if self.category_cols:
            for col, size, choices in zip(self.category_cols, self.category_sizes, self.encoder.categories_):
                block = values[:, start:start + size]
                result[col] = choices[block.argmax(axis=1)]
                start += size
        return pd.DataFrame(result, columns=self.numeric_cols + self.category_cols)
