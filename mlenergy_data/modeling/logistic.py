"""Four-parameter logistic fit model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LogisticModel:
    """Four-parameter logistic: `y = b0 + L * sigmoid(k * (x - x0))`.

    `x` is typically `log2(batch_size)`.

    Attributes:
        L: Amplitude (upper asymptote minus lower asymptote).
        x0: Midpoint of the sigmoid on the x-axis.
        k: Steepness of the sigmoid curve.
        b0: Baseline offset (lower asymptote).
    """

    L: float
    x0: float
    k: float
    b0: float

    def eval_x(self, x: float) -> float:
        """Evaluate at continuous x (= log2(batch_size))."""
        a = self.k * (float(x) - self.x0)
        if a >= 0:
            ea = math.exp(-a)
            s = 1.0 / (1.0 + ea)
        else:
            ea = math.exp(a)
            s = ea / (1.0 + ea)
        return float(self.b0 + self.L * s)

    def deriv_wrt_x(self, x: float) -> float:
        """dy/dx for y = b0 + L * sigmoid(k*(x - x0))."""
        a = self.k * (float(x) - self.x0)
        if a >= 0:
            ea = math.exp(-a)
            s = 1.0 / (1.0 + ea)
        else:
            ea = math.exp(a)
            s = ea / (1.0 + ea)
        ds_dx = self.k * s * (1.0 - s)
        return float(self.L * ds_dx)

    def eval(self, batch: int) -> float:
        """Evaluate at an integer batch size (converted to log2)."""
        return self.eval_x(math.log2(max(int(batch), 1)))

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray) -> LogisticModel:
        """Fit four-parameter logistic using coarse grid search + least-squares.

        Args:
            x: Independent variable (typically log2(batch_size)).
            y: Dependent variable (e.g., power, latency, throughput).

        Returns:
            Fitted LogisticModel instance.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.size == 0 or y.size == 0:
            return cls(L=0.0, x0=0.0, k=1.0, b0=0.0)
        if x.size == 1 or np.allclose(y, y[0]):
            return cls(L=0.0, x0=float(x[0]), k=1.0, b0=float(y.mean()))

        x0_grid = np.linspace(float(x.min()) - 1.0, float(x.max()) + 1.0, 40)
        k_grid = np.logspace(-2, 2, 60)

        best = (float("inf"), 0.0, float(x.mean()), 1.0, float(y.mean()))
        ones = np.ones_like(x)

        for x0_ in x0_grid:
            z = x - float(x0_)
            for k_ in k_grid:
                s = 1.0 / (1.0 + np.exp(-float(k_) * z))
                A = np.column_stack((ones, s))
                coeff, *_ = np.linalg.lstsq(A, y, rcond=None)
                b0_ = float(coeff[0])
                L_ = float(coeff[1])
                yhat = b0_ + L_ * s
                mse = float(np.mean((y - yhat) ** 2))
                if mse < best[0]:
                    best = (mse, L_, float(x0_), float(k_), b0_)

        _, L, x0_, k_, b0_ = best
        return cls(L=L, x0=x0_, k=k_, b0=b0_)

    def to_dict(self) -> dict[str, float]:
        """Serialize parameters to a dict."""
        return {"L": self.L, "x0": self.x0, "k": self.k, "b0": self.b0}

    @classmethod
    def from_dict(cls, d: dict[str, Any] | Any) -> LogisticModel:
        """Deserialize parameters from a dict (or dict-like, e.g. pandas Series).

        Args:
            d: Dict with keys `L`, `x0`, `k`, `b0`.
        """
        return cls(L=float(d["L"]), x0=float(d["x0"]), k=float(d["k"]), b0=float(d["b0"]))
