"""Trainable attribution and SIFT readouts with frozen thresholds and SGR abstention."""

from __future__ import annotations

import numpy as np

from .metrics import auroc, sgr_threshold


class _Standardizer:
    def fit(self, values: np.ndarray):
        self.mean = values.mean(axis=0)
        self.scale = values.std(axis=0)
        self.scale[self.scale < 1e-6] = 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.clip((values - self.mean) / self.scale, -8, 8)


class Detector:
    """A compact logistic readout; SIFT optionally adds IRMv1 and abstention."""

    name = "detector"
    has_abstention = False

    def __init__(self, *, seed: int = 0, hidden: int = 0, irm_lambda: float = 0.0, epochs: int = 400, learning_rate: float = 3e-3, weight_decay: float = 1e-3):
        self.seed, self.hidden, self.irm_lambda = seed, hidden, irm_lambda
        self.epochs, self.learning_rate, self.weight_decay, self.threshold = epochs, learning_rate, weight_decay, 0.5

    def fit(self, features: np.ndarray, labels: np.ndarray, environments: np.ndarray | None = None, device: str = "cpu"):
        try:
            import torch
            import torch.nn.functional as functional
        except ImportError as exc:
            raise RuntimeError("Detector fitting requires PyTorch.") from exc
        features, labels = np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.float32)
        if features.ndim != 2 or features.shape[0] != labels.size or len(np.unique(labels)) < 2:
            raise ValueError("features and binary labels must contain both classes")
        torch.manual_seed(self.seed)
        self.standardizer = _Standardizer().fit(features)
        x = torch.tensor(self.standardizer.transform(features), device=device)
        y = torch.tensor(labels, device=device)
        layers = [torch.nn.Linear(x.shape[1], self.hidden), torch.nn.ReLU(), torch.nn.Linear(self.hidden, 1)] if self.hidden else [torch.nn.Linear(x.shape[1], 1)]
        self.model = torch.nn.Sequential(*layers).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        env = np.zeros(labels.size, dtype=int) if environments is None else np.asarray(environments)
        groups = [np.flatnonzero(env == value) for value in np.unique(env) if np.sum(env == value) >= 4]
        positive_weight = float((labels == 0).sum() / max((labels == 1).sum(), 1))
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            logits = self.model(x).squeeze(-1)
            loss = functional.binary_cross_entropy_with_logits(logits, y, pos_weight=torch.tensor(positive_weight, device=device))
            if self.irm_lambda and len(groups) > 1:
                penalties = []
                for indices in groups:
                    scale = torch.ones(1, requires_grad=True, device=device)
                    risk = functional.binary_cross_entropy_with_logits(logits[indices] * scale, y[indices])
                    penalties.append(torch.autograd.grad(risk, scale, create_graph=True)[0].square().sum())
                loss = loss + (self.irm_lambda if epoch > self.epochs // 4 else 1.0) * torch.stack(penalties).mean()
            loss.backward()
            optimizer.step()
        self.device = device
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            values = torch.tensor(self.standardizer.transform(np.asarray(features, dtype=np.float32)), device=self.device)
            return torch.sigmoid(self.model(values).squeeze(-1)).cpu().numpy()

    def freeze_threshold(self, validation_features: np.ndarray, validation_labels: np.ndarray) -> float:
        scores, labels = self.score(validation_features), np.asarray(validation_labels)
        best, self.threshold = -2.0, 0.5
        for threshold in np.unique(np.round(scores, 4)):
            true_positive = np.mean(scores[labels == 1] > threshold) if np.any(labels == 1) else 0.0
            false_positive = np.mean(scores[labels == 0] > threshold) if np.any(labels == 0) else 0.0
            if true_positive - false_positive > best:
                best, self.threshold = true_positive - false_positive, float(threshold)
        return self.threshold

    def verdict(self, features: np.ndarray) -> np.ndarray:
        return (self.score(features) > self.threshold).astype(int)

    def validation_auroc(self, features: np.ndarray, labels: np.ndarray) -> float:
        return auroc(self.score(features), labels)


class AttributionConsistency(Detector):
    name = "attribution"


class SIFT(Detector):
    name, has_abstention = "sift", True

    def __init__(self, *, use_irm: bool = True, use_abstention: bool = True, **kwargs):
        super().__init__(hidden=kwargs.pop("hidden", 32), irm_lambda=kwargs.pop("irm_lambda", 100.0) if use_irm else 0.0, **kwargs)
        self.use_abstention, self.gate_threshold, self.coverage = use_abstention, float("-inf"), 1.0

    def gate(self, features: np.ndarray) -> np.ndarray:
        scores = self.score(features)
        return np.where(scores > self.threshold, (scores - self.threshold) / max(1 - self.threshold, 1e-6), (self.threshold - scores) / max(self.threshold, 1e-6))

    def calibrate(self, features: np.ndarray, labels: np.ndarray, alpha: float = 0.25, delta: float = 0.05) -> float:
        if not self.use_abstention:
            return self.gate_threshold
        self.gate_threshold, self.coverage, self.calibration_risk, self.calibration_bound = sgr_threshold(self.gate(features), self.verdict(features) == np.asarray(labels), alpha, delta)
        return self.gate_threshold

    def certified(self, features: np.ndarray) -> np.ndarray:
        return self.gate(features) >= self.gate_threshold
