"""Evidence extraction: which measurable acoustic changes support each attributed cause."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ..features.extractor import describe, group_of
from .model import CausalAttributor


@dataclass
class Evidence:
    feature: str
    description: str
    unit: str
    group: str
    observed: float
    baseline: float
    z: float
    contribution: float  # drop in P(cause) when this feature is set to its baseline

    def sentence(self) -> str:
        direction = "increased" if self.z > 0 else "decreased"
        is_delta = self.feature.startswith("delta::")
        what = f"{self.description} {direction}" if is_delta else f"{self.description} is {'high' if self.z > 0 else 'low'}"
        val = f" by {abs(self.observed):.3g} {self.unit}".rstrip() if is_delta else f" ({self.observed:.3g} {self.unit})".replace(" )", ")")
        return f"{what}{val} (z={self.z:+.1f}, contributes {self.contribution:+.2f} to the probability)"


@dataclass
class CauseReport:
    cause: str
    probability: float
    present: bool
    evidence: list[Evidence] = field(default_factory=list)
    group_contribution: dict[str, float] = field(default_factory=dict)


@dataclass
class AttributionResult:
    causes: list[CauseReport]
    novelty: float
    uncertainty: float
    novelty_threshold: float = 1.0

    @property
    def is_unknown(self) -> bool:
        return self.novelty > self.novelty_threshold

    def present(self) -> list[str]:
        return [c.cause for c in self.causes if c.present]

    def to_dict(self) -> dict:
        return {"present": self.present(), "novelty": self.novelty, "uncertainty": self.uncertainty,
                "is_unknown": self.is_unknown, "causes": [asdict(c) for c in self.causes]}

    def text(self, top_causes: int = 3) -> str:
        lines = []
        if self.is_unknown:
            lines.append(f"WARNING: pattern outside the known cause space (novelty {self.novelty:.2f} > 1). "
                         "Attributions below may be unreliable; possible unknown manipulation.")
        ranked = sorted(self.causes, key=lambda c: -c.probability)
        shown = [c for c in ranked if c.present] or ranked[:1]
        if not any(c.present for c in self.causes):
            lines.append("No known cause exceeds the decision threshold (closest candidate shown).")
        for c in shown[:top_causes]:
            lines.append(f"- {c.cause}: probability {c.probability:.2f}")
            for ev in c.evidence:
                lines.append(f"    * {ev.sentence()}")
        return "\n".join(lines)


def occlusion_importance(att: CausalAttributor, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature occlusion importance for every cause in a single batched model call.

    Returns (p0, imp) where p0 is the probability vector and imp[j, c] = p0[c] - p_c(x with
    feature j replaced by its no-change baseline).
    """
    x = np.asarray(x, float)
    n = len(x)
    batch = np.tile(x, (n + 1, 1))
    for j in range(n):
        batch[j + 1, j] = att.baseline_[j]
    P = att.predict_proba(batch)
    return P[0], P[0][None, :] - P[1:]


def group_importance(att: CausalAttributor, x: np.ndarray) -> dict[str, np.ndarray]:
    """Occlude whole feature groups (e.g. all formant features) and return the drop per cause."""
    groups: dict[str, list[int]] = {}
    for j, name in enumerate(att.feature_names):
        groups.setdefault(group_of(name), []).append(j)
    names = list(groups)
    batch = np.tile(np.asarray(x, float), (len(names) + 1, 1))
    for gi, g in enumerate(names):
        batch[gi + 1, groups[g]] = att.baseline_[groups[g]]
    P = att.predict_proba(batch)
    return {g: P[0] - P[i + 1] for i, g in enumerate(names)}


def explain(att: CausalAttributor, x: np.ndarray, top_k: int = 5) -> AttributionResult:
    """Full explainable attribution for one feature vector."""
    x = np.asarray(x, float)
    p0, imp = occlusion_importance(att, x)
    gimp = group_importance(att, x)
    z = (x - att.baseline_) / att.baseline_mad_
    reports = []
    for ci, cause in enumerate(att.causes):
        order = np.argsort(-imp[:, ci])[:top_k]
        evid = [Evidence(feature=att.feature_names[j], description=describe(att.feature_names[j])[0],
                         unit=describe(att.feature_names[j])[1], group=group_of(att.feature_names[j]),
                         observed=float(x[j]), baseline=float(att.baseline_[j]),
                         z=float(np.clip(z[j], -99, 99)), contribution=float(imp[j, ci]))
                for j in order if imp[j, ci] > 1e-4]
        reports.append(CauseReport(cause=cause, probability=float(p0[ci]),
                                   present=bool(p0[ci] >= att.threshold), evidence=evid,
                                   group_contribution={g: float(v[ci]) for g, v in gimp.items()}))
    return AttributionResult(causes=reports, novelty=float(att.novelty(x)[0]),
                             uncertainty=float(att.uncertainty(x)[0]))
