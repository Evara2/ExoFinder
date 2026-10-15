import json
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score
)

sys.path.insert(0, str(Path("src").resolve()))
from train import ExoNet, prepare

torch.set_num_threads(4)

df = pd.read_csv("data/exoTest.csv")
X, y = prepare(df)

checkpoint = torch.load(
    "models/exofinder.pt",
    map_location="cpu",
    weights_only=False
)

model = ExoNet(input_length=3197)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

with torch.no_grad():
    scores = torch.sigmoid(
        model(torch.from_numpy(X))
    ).numpy()

with open("outputs/training_metrics.json") as f:
    threshold = json.load(f)["threshold"]

predictions = (scores >= threshold).astype(int)

print("TEST RESULTS")
print("Threshold:", threshold)
print("PR-AUC:", average_precision_score(y, scores))
print("ROC-AUC:", roc_auc_score(y, scores))
print("Confusion Matrix:")
print(confusion_matrix(y, predictions))
print(classification_report(
    y, predictions,
    target_names=["Non-exoplanet", "Exoplanet"],
    zero_division=0
))
