"""Online trade outcome model for win-probability estimation."""

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np


@dataclass
class OnlineTradeOutcomeModel:
    n_features: int
    learning_rate: float = 0.02
    l2: float = 1e-4

    def __post_init__(self):
        self.weights = np.zeros(self.n_features, dtype=np.float32)
        self.bias = 0.0
        self.samples = 0
        self.wins = 0

    def _sanitize(self, x: np.ndarray) -> np.ndarray:
        arr = np.array(x, dtype=np.float32).reshape(-1)
        if arr.shape[0] != self.n_features:
            fixed = np.zeros(self.n_features, dtype=np.float32)
            n = min(self.n_features, arr.shape[0])
            if n > 0:
                fixed[:n] = arr[:n]
            arr = fixed
        arr = np.nan_to_num(arr, nan=0.0, posinf=5.0, neginf=-5.0)
        return np.clip(arr, -5.0, 5.0)

    @staticmethod
    def _sigmoid(z: float) -> float:
        z = float(np.clip(z, -30.0, 30.0))
        return float(1.0 / (1.0 + np.exp(-z)))

    def predict_proba(self, x: np.ndarray) -> float:
        x = self._sanitize(x)
        return self._sigmoid(np.dot(self.weights, x) + self.bias)

    def update(self, x: np.ndarray, win_label: int, sample_weight: float = 1.0):
        x = self._sanitize(x)
        y = 1.0 if win_label else 0.0
        p = self.predict_proba(x)
        err = (p - y) * max(0.2, min(2.0, float(sample_weight)))
        self.weights -= self.learning_rate * (err * x + self.l2 * self.weights)
        self.bias -= self.learning_rate * err
        self.samples += 1
        self.wins += int(y)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "samples": self.samples,
            "win_rate": round(self.wins / max(1, self.samples), 4),
            "weight_magnitude": round(float(np.mean(np.abs(self.weights))), 5),
        }

    def export(self) -> Dict[str, Any]:
        return {
            "n_features": self.n_features,
            "learning_rate": self.learning_rate,
            "l2": self.l2,
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "samples": self.samples,
            "wins": self.wins,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], n_features: int):
        model = cls(n_features=n_features,
                    learning_rate=float(data.get("learning_rate", 0.02)),
                    l2=float(data.get("l2", 1e-4)))
        weights = np.array(data.get("weights", []), dtype=np.float32)
        if weights.shape[0] == n_features:
            model.weights = weights
        model.bias = float(data.get("bias", 0.0))
        model.samples = max(0, int(data.get("samples", 0)))
        model.wins = max(0, int(data.get("wins", 0)))
        return model
