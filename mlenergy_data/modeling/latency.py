"""Two-component lognormal mixture model for inter-token latency."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ITLMixtureModel:
    """Two-component ITL mixture where each component is log-normal plus offset.

    Attributes:
        loc: Location shift applied to all samples.
        pi_steady: Mixture weight for the steady (low-latency) component.
        sigma_steady: Log-normal sigma for the steady component.
        scale_steady: Log-normal scale for the steady component.
        pi_stall: Mixture weight for the stall (high-latency) component.
        sigma_stall: Log-normal sigma for the stall component.
        scale_stall: Log-normal scale for the stall component.
    """

    loc: float
    pi_steady: float
    sigma_steady: float
    scale_steady: float
    pi_stall: float
    sigma_stall: float
    scale_stall: float

    def _lognormal_mean_var(self, sigma: float, scale: float) -> tuple[float, float]:
        s2 = float(sigma) ** 2
        ey = float(scale) * math.exp(0.5 * s2)
        vy = (float(scale) ** 2) * (math.exp(s2) - 1.0) * math.exp(s2)
        return ey, vy

    def mean_var(self) -> tuple[float, float]:
        """Compute mean and variance of the two-component mixture."""
        p1 = float(self.pi_steady)
        p2 = float(self.pi_stall)
        ps = max(p1 + p2, 1e-12)
        p1 /= ps
        p2 /= ps

        m1, v1 = self._lognormal_mean_var(self.sigma_steady, self.scale_steady)
        m2, v2 = self._lognormal_mean_var(self.sigma_stall, self.scale_stall)

        m1x = float(self.loc) + m1
        m2x = float(self.loc) + m2
        mx = p1 * m1x + p2 * m2x
        ex2 = p1 * (v1 + m1x * m1x) + p2 * (v2 + m2x * m2x)
        vx = max(ex2 - mx * mx, 0.0)
        return mx, vx

    def sample_one(self, rng: np.random.Generator) -> float:
        """Draw a single sample from the mixture."""
        p1 = float(self.pi_steady)
        p2 = float(self.pi_stall)
        ps = max(p1 + p2, 1e-12)
        p1 /= ps
        if rng.random() < p1:
            y = rng.lognormal(
                mean=math.log(max(self.scale_steady, 1e-15)),
                sigma=max(self.sigma_steady, 0.0),
            )
        else:
            y = rng.lognormal(
                mean=math.log(max(self.scale_stall, 1e-15)),
                sigma=max(self.sigma_stall, 0.0),
            )
        return float(self.loc + y)

    def sample_avg(
        self,
        *,
        n_replicas: int,
        rng: np.random.Generator,
        exact_threshold: int = 30,
    ) -> float:
        """Sample the average ITL across n_replicas replicas.

        For n <= exact_threshold: draw n individual samples and average.
        For n > exact_threshold: use CLT approximation (normal from mean/var).

        Args:
            n_replicas: Number of replicas to average over.
            rng: NumPy random generator.
            exact_threshold: Below this count, use exact sampling.

        Returns:
            Average ITL in seconds.  NaN if n_replicas <= 0.
        """
        n = int(n_replicas)
        if n <= 0:
            return float("nan")

        if n <= int(exact_threshold):
            vals = [self.sample_one(rng) for _ in range(n)]
            return float(np.mean(vals))

        mu, var = self.mean_var()
        sd = math.sqrt(max(var / float(n), 0.0))
        x = float(rng.normal(mu, sd))
        return float(max(x, 0.0))

    @classmethod
    def fit(
        cls,
        samples_s: np.ndarray,
        *,
        max_samples: int | None = None,
        seed: int = 0,
    ) -> ITLMixtureModel:
        """Fit a two-component lognormal mixture from ITL samples.

        Args:
            samples_s: Raw ITL samples in seconds.
            max_samples: If set, randomly subsample to this many values
                before fitting.
            seed: RNG seed for subsampling.

        Returns:
            Fitted ITLMixtureModel instance.
        """
        x = np.asarray(samples_s, dtype=float)
        x = x[np.isfinite(x) & (x > 0.0)]

        if max_samples is not None and x.size > max_samples:
            rng = np.random.default_rng(seed)
            idx = rng.choice(x.size, size=max_samples, replace=False)
            x = x[idx]

        if x.size == 0:
            return cls(
                loc=0.0,
                pi_steady=1.0,
                sigma_steady=0.1,
                scale_steady=1e-3,
                pi_stall=0.0,
                sigma_stall=0.1,
                scale_stall=2e-3,
            )

        loc = max(0.0, float(np.min(x)) - 1e-6)
        z = np.log(np.maximum(x - loc, 1e-12))
        if z.size < 5 or float(np.std(z)) < 1e-6:
            m = float(np.mean(z))
            s = max(float(np.std(z)), 0.1)
            return cls(
                loc=loc,
                pi_steady=1.0,
                sigma_steady=s,
                scale_steady=float(math.exp(m)),
                pi_stall=0.0,
                sigma_stall=s,
                scale_stall=float(math.exp(m + s)),
            )

        mu1 = float(np.quantile(z, 0.35))
        mu2 = float(np.quantile(z, 0.85))
        sd = float(np.std(z))
        s1 = max(sd * 0.5, 0.1)
        s2 = max(sd, 0.1)
        pi1 = 0.8
        pi2 = 0.2

        for _ in range(80):
            p1 = pi1 * np.exp(-0.5 * ((z - mu1) / s1) ** 2) / (s1 + 1e-12)
            p2 = pi2 * np.exp(-0.5 * ((z - mu2) / s2) ** 2) / (s2 + 1e-12)
            den = p1 + p2 + 1e-12
            r1 = p1 / den
            r2 = p2 / den

            w1 = float(np.sum(r1))
            w2 = float(np.sum(r2))
            if w1 <= 1e-9 or w2 <= 1e-9:
                break
            pi1 = w1 / float(z.size)
            pi2 = 1.0 - pi1
            mu1 = float(np.sum(r1 * z) / w1)
            mu2 = float(np.sum(r2 * z) / w2)
            s1 = max(float(np.sqrt(np.sum(r1 * (z - mu1) ** 2) / w1)), 0.05)
            s2 = max(float(np.sqrt(np.sum(r2 * (z - mu2) ** 2) / w2)), 0.05)

        if mu1 <= mu2:
            return cls(
                loc=loc,
                pi_steady=float(pi1),
                sigma_steady=float(s1),
                scale_steady=float(math.exp(mu1)),
                pi_stall=float(pi2),
                sigma_stall=float(s2),
                scale_stall=float(math.exp(mu2)),
            )

        return cls(
            loc=loc,
            pi_steady=float(pi2),
            sigma_steady=float(s2),
            scale_steady=float(math.exp(mu2)),
            pi_stall=float(pi1),
            sigma_stall=float(s1),
            scale_stall=float(math.exp(mu1)),
        )

    def to_dict(self) -> dict[str, float]:
        """Serialize parameters to a dict."""
        return {
            "loc": self.loc,
            "pi_steady": self.pi_steady,
            "sigma_steady": self.sigma_steady,
            "scale_steady": self.scale_steady,
            "pi_stall": self.pi_stall,
            "sigma_stall": self.sigma_stall,
            "scale_stall": self.scale_stall,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | Any) -> ITLMixtureModel:
        """Deserialize parameters from a dict (or dict-like, e.g. pandas Series).

        The dict keys may optionally use the `itl_mix_` prefix (e.g., from
        CSV files produced by the build pipeline).

        Args:
            d: Dict with keys for all 7 mixture parameters.
        """

        def _get(key: str) -> float:
            if key in d:
                return float(d[key])
            return float(d[f"itl_mix_{key}"])

        return cls(
            loc=_get("loc"),
            pi_steady=_get("pi_steady"),
            sigma_steady=_get("sigma_steady"),
            scale_steady=_get("scale_steady"),
            pi_stall=_get("pi_stall"),
            sigma_stall=_get("sigma_stall"),
            scale_stall=_get("scale_stall"),
        )
