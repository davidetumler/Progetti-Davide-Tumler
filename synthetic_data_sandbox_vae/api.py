"""API locale FastAPI. NON esporre su Internet o reti condivise."""
from __future__ import annotations

from io import BytesIO
from typing import Literal
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from analysis_engine import analyze_synthetic
from service import sandbox, MAX_UPLOAD_BYTES

app = FastAPI(title='Synthetic Data Sandbox API', version='1.0')


class TrainRequest(BaseModel):
    epochs: int = Field(default=25, ge=1, le=150)
    latent_dim: int = Field(default=4, ge=2, le=32)


class GenerateRequest(BaseModel):
    rows: int = Field(default=300, ge=20, le=10000)


class AnalyzeRequest(BaseModel):
    question: str = Field(min_length=3, max_length=700)


def _bad_request(exc: ValueError) -> None:
    raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get('/status')
def status():
    return sandbox.status()


@app.post('/upload')
async def upload(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith('.csv'):
        raise HTTPException(400, detail='Carica un file .csv')
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    try:
        return sandbox.upload(content)
    except ValueError as exc:
        _bad_request(exc)


@app.post('/train')
def train(request: TrainRequest):
    try:
        return sandbox.train(epochs=request.epochs, latent_dim=request.latent_dim)
    except ValueError as exc:
        _bad_request(exc)


@app.post('/generate')
def generate(request: GenerateRequest):
    try:
        return sandbox.generate(rows=request.rows)
    except ValueError as exc:
        _bad_request(exc)


@app.get('/preview')
def preview(source: Literal['real', 'synthetic'] = 'synthetic', rows: int = 10):
    if not 1 <= rows <= 100:
        raise HTTPException(400, detail='rows deve essere tra 1 e 100')
    try:
        return sandbox.preview(source, rows)
    except ValueError as exc:
        _bad_request(exc)


@app.get('/validation')
def validation():
    try:
        return sandbox.validation()
    except ValueError as exc:
        _bad_request(exc)


@app.post('/analyze')
def analyze(request: AnalyzeRequest):
    try:
        if sandbox.synthetic is None:
            raise ValueError('Devi prima generare dati sintetici.')
        return analyze_synthetic(request.question, sandbox.synthetic)
    except ValueError as exc:
        _bad_request(exc)


@app.get('/download')
def download():
    if sandbox.synthetic is None:
        raise HTTPException(400, detail='Genera prima i dati sintetici.')
    content = sandbox.synthetic.to_csv(index=False).encode('utf-8')
    return StreamingResponse(BytesIO(content), media_type='text/csv',
                             headers={'Content-Disposition': 'attachment; filename="synthetic_data.csv"'})
