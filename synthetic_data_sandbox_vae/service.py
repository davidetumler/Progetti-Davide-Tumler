"""Logica condivisa per API, statistiche e controllo privacy di base."""
from __future__ import annotations

from io import BytesIO
import numpy as np
import pandas as pd
from data_engine import DataEngine
from vae_model import TabularVAE, train_vae

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class SandboxService:
    def __init__(self):
        self.original: pd.DataFrame | None = None
        self.synthetic: pd.DataFrame | None = None
        self.engine: DataEngine | None = None
        self.model: TabularVAE | None = None
        self.history: list[float] = []

    def upload(self, content: bytes) -> dict:
        if not content or len(content) > MAX_UPLOAD_BYTES:
            raise ValueError('File mancante o superiore a 10 MB.')
        try:
            df = pd.read_csv(BytesIO(content))
        except Exception as exc:
            raise ValueError('Il file non è un CSV leggibile (preferisci UTF-8).') from exc
        if len(df) > 10000:
            raise ValueError('Per questa demo sono ammesse al massimo 10.000 righe.')
        engine = DataEngine()
        engine.fit_transform(df)
        self.original = engine.clean(df)
        self.engine = engine
        self.model = None
        self.synthetic = None
        self.history = []
        return {'rows': len(df), 'columns': df.columns.tolist(), 'features': engine.feature_count}

    def train(self, epochs: int, latent_dim: int) -> dict:
        if self.original is None or self.engine is None:
            raise ValueError('Carica prima un dataset.')
        x = self.engine.transform(self.original)
        model, history = train_vae(x, latent_dim=latent_dim, epochs=epochs)
        self.model, self.history = model, history
        self.synthetic = None
        return {'epochs': epochs, 'latent_dim': latent_dim, 'loss': history}

    def generate(self, rows: int) -> dict:
        if self.model is None:
            raise ValueError('Addestra prima il VAE.')
        self.synthetic = self.engine.inverse_transform(self.model.sample(rows))
        return {'rows': len(self.synthetic), 'columns': self.synthetic.columns.tolist()}

    def status(self) -> dict:
        return {'uploaded': self.original is not None, 'trained': self.model is not None,
                'generated': self.synthetic is not None,
                'rows_real': len(self.original) if self.original is not None else 0,
                'rows_synthetic': len(self.synthetic) if self.synthetic is not None else 0}

    def preview(self, source: str, rows: int) -> dict:
        df = self.original if source == 'real' else self.synthetic
        if df is None:
            raise ValueError('Non ci sono dati disponibili per questa vista.')
        # JSON compatibile: nessun NaN e nessun tipo numpy
        return {'columns': df.columns.tolist(), 'records': df.head(rows).to_json(orient='records'),
                'rows_total': len(df)}

    def validation(self) -> dict:
        if self.original is None or self.synthetic is None:
            raise ValueError('Servono dati originali e sintetici.')
        real, fake = self.original, self.synthetic
        nums = self.engine.numeric_cols
        diff = None
        if len(nums) > 1:
            a = real[nums].corr().to_numpy()
            b = fake[nums].corr().to_numpy()
            valid = np.isfinite(a) & np.isfinite(b)
            diff = round(float(np.mean(np.abs(a[valid] - b[valid]))), 4) if valid.any() else None
        # Confronto approssimato: NON è una garanzia di privacy differenziale.
        real_keys = set(map(tuple, real.round(4).astype(str).to_numpy()))
        matches = sum(tuple(row) in real_keys for row in fake.round(4).astype(str).to_numpy())
        return {'correlation_mean_absolute_error': diff, 'exact_matches_rounded': matches,
                'real_rows': len(real), 'synthetic_rows': len(fake),
                'warning': 'Un basso numero di corrispondenze NON prova che i dati siano anonimi. '
                           'Non usare questa demo come garanzia di privacy su dati veri.'}


sandbox = SandboxService()
