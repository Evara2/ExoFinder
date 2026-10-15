import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from train import ExoNet, prepare

app = FastAPI(title='ExoFinder API', version='1.0')
torch.set_num_threads(min(4, torch.get_num_threads()))
checkpoint = torch.load(ROOT / 'models' / 'exofinder.pt', map_location='cpu', weights_only=True)
if checkpoint.get('architecture') != 'ExoNet-v1':
    raise RuntimeError('Unexpected checkpoint architecture')
input_length = int(checkpoint['input_length'])
model = ExoNet(input_length=input_length)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
threshold = float(json.loads((ROOT / 'outputs' / 'training_metrics.json').read_text())['threshold'])

class Curve(BaseModel):
    flux: list[float] = Field(min_length=input_length, max_length=input_length)

def infer(df):
    if len(df) == 0 or len(df) > 1000:
        raise HTTPException(400, 'Upload between 1 and 1000 light curves')
    if df.shape[1] != input_length + 1:
        raise HTTPException(400, f'Expected LABEL and {input_length} FLUX columns')
    x, _ = prepare(df)
    with torch.inference_mode():
        scores = torch.sigmoid(model(torch.from_numpy(x))).cpu().numpy().tolist()
    return [{'index': i, 'probability': round(float(p), 6), 'candidate': bool(p >= threshold)} for i, p in enumerate(scores)]

@app.get('/health')
def health():
    return {'status': 'ok', 'model': checkpoint['architecture']}

@app.get('/model-info')
def model_info():
    return {'architecture': checkpoint['architecture'], 'input_length': input_length, 'threshold': threshold}

@app.post('/predict')
def predict(curve: Curve):
    df = pd.DataFrame([curve.flux], columns=[f'FLUX.{i}' for i in range(1, input_length + 1)])
    df.insert(0, 'LABEL', 1)
    return infer(df)[0]

@app.post('/predict-csv')
async def predict_csv(file: UploadFile = File(...)):
    MAX_UPLOAD_SIZE = 50 * 1024 * 1024

    data = await file.read(MAX_UPLOAD_SIZE + 1)

    if len(data) > MAX_UPLOAD_SIZE:
        raise HTTPException(413, 'CSV exceeds 50 MB')
    try:
        df = pd.read_csv(io.BytesIO(data))
        expected = ['LABEL'] + [f'FLUX.{i}' for i in range(1, input_length + 1)]
        if list(df.columns) != expected:
            raise ValueError('Expected LABEL followed by FLUX.1 ... FLUX.3197')
        return {'count': len(df), 'threshold': threshold, 'predictions': infer(df)}
    except (ValueError, pd.errors.ParserError) as exc:
        raise HTTPException(400, str(exc)) from exc
