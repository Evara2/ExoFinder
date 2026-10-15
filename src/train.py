import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class ExoNet(nn.Module):
    def __init__(self, input_length=3197):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 9, stride=4, padding=4), nn.BatchNorm1d(16), nn.GELU(),
            nn.Conv1d(16, 32, 7, stride=4, padding=3), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=4, padding=2), nn.BatchNorm1d(64), nn.GELU(),
        )
        self.position = nn.Parameter(torch.zeros(1, 50, 64))
        layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128,
                                           dropout=0.1, batch_first=True, norm_first=False)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 32), nn.GELU(), nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, x):
        x = self.features(x.unsqueeze(1)).transpose(1, 2)
        x = self.transformer(x + self.position[:, :x.shape[1]])
        return self.head(x.mean(dim=1)).squeeze(1)


def prepare(df, scaler=None, fit=False):
    y = (df['LABEL'].to_numpy() == 2).astype(np.float32)
    x = df.drop(columns=['LABEL']).to_numpy(dtype=np.float32)
    x[~np.isfinite(x)] = np.nan
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    # Normalize each light curve independently; this is also used during inference.
    med = np.median(x, axis=1, keepdims=True)
    q25 = np.percentile(x, 25, axis=1, keepdims=True)
    q75 = np.percentile(x, 75, axis=1, keepdims=True)
    scale = np.maximum(q75 - q25, 1e-6)
    x = np.clip((x - med) / scale, -10, 10).astype(np.float32)
    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--data', default='data/exoTrain.csv')
    parser.add_argument('--output', default='models/exofinder.pt')
    args = parser.parse_args()
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    df = pd.read_csv(args.data)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['LABEL'], random_state=42)
    xtr, ytr = prepare(train_df); xval, yval = prepare(val_df)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr)),
                              batch_size=args.batch_size, shuffle=True)
    val_x = torch.from_numpy(xval)
    model = ExoNet()
    positive_weight = torch.tensor([min(100.0, (len(ytr) - ytr.sum()) / max(1, ytr.sum()))])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    best_score, best_state, best_epoch = -1.0, None, 0
    for epoch in range(args.epochs):
        model.train(); running = 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            running += loss.item() * len(yb)
        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(model(val_x)).numpy()
        ap = average_precision_score(yval, scores)
        print(f'Epoch {epoch+1:02d}: train_loss={running/len(ytr):.4f} val_PR_AUC={ap:.4f}', flush=True)
        if ap > best_score:
            best_score, best_state, best_epoch = ap, copy.deepcopy(model.state_dict()), epoch+1
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': best_state, 'architecture': 'ExoNet-v1',
                'input_length': 3197, 'best_epoch': best_epoch,
                'validation_pr_auc': float(best_score)}, args.output)
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(val_x)).numpy()
    # Threshold selection uses validation data only; test remains untouched.
    thresholds = np.linspace(0.05, 0.95, 91)
    f1s = []
    for t in thresholds:
        tp = ((scores >= t) & (yval == 1)).sum()
        fp = ((scores >= t) & (yval == 0)).sum()
        fn = ((scores < t) & (yval == 1)).sum()
        f1s.append(float(2 * tp / max(1, 2*tp + fp + fn)))
    threshold = float(thresholds[int(np.argmax(f1s))])
    metadata = {'threshold': threshold, 'validation_pr_auc': float(best_score),
                'validation_f1': float(max(f1s)), 'best_epoch': best_epoch,
                'train_samples': len(ytr), 'validation_samples': len(yval),
                'validation_positive_samples': int(yval.sum())}
    Path('outputs').mkdir(exist_ok=True)
    Path('outputs/training_metrics.json').write_text(json.dumps(metadata, indent=2))
    print('Saved:', args.output, 'Metadata:', 'outputs/training_metrics.json')
    print('Validation confusion matrix:', confusion_matrix(yval, scores >= threshold).tolist())


if __name__ == '__main__':
    main()
