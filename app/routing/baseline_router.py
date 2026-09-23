"""The originally-proposed router: per-model success classifiers trained
offline on either handcrafted features alone, handcrafted features plus the
embedding, or the embedding alone — this is the "learned router" the kNN
design is benchmarked against. Same cheapest-good-enough decision rule as
KnnRouter so the comparison isolates the estimator, not the decision logic.
"""

import time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from app.routing.model_registry import model_registry as _default_registry
from app.routing.features import extract_features
from app.routing.knn_router import RouteDecision


class _ConstantClassifier:
    """Fallback for a model whose training labels are all one class (e.g.
    always succeeds, or never appears) — sklearn classifiers can't fit on
    a single class, so skip fitting and just predict that constant rate."""
    def __init__(self, p: float):
        self.p = p

    def predict_proba(self, X):
        n = X.shape[0]
        return np.tile([1 - self.p, self.p], (n, 1))


def _predict_p1(clf, X) -> float:
    proba = clf.predict_proba(X)
    if proba.shape[1] == 1:
        return float(proba[0, 0])
    return float(proba[0, 1])


def _fit_one(clf_factory, X: np.ndarray, y: np.ndarray):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return _ConstantClassifier(float(y.mean()) if len(y) else 0.5)
    clf = clf_factory()
    clf.fit(X, y)
    return clf


class _BaseLearnedRouter:
    """Shared fit/route machinery; subclasses only define how a prompt +
    embedding become a feature vector."""
    name = "base_learned"

    def __init__(self, registry=None, quality_target: float = 0.8):
        self.registry = registry or _default_registry
        self.quality_target = quality_target
        self.models = {}   # model_name -> fitted classifier

    def _clf_factory(self):
        raise NotImplementedError

    def _featurize(self, prompt: str, embedding) -> np.ndarray:
        raise NotImplementedError

    def fit(self, prompts, embeddings, labels_per_model: dict):
        """labels_per_model: {model_name: [0/1, ...]} aligned with prompts."""
        X = np.stack([self._featurize(p, e) for p, e in zip(prompts, embeddings)])
        for m_name, y in labels_per_model.items():
            self.models[m_name] = _fit_one(self._clf_factory, X, y)

    def _p_hat(self, prompt: str, embedding) -> dict:
        X = self._featurize(prompt, embedding).reshape(1, -1)
        return {m_name: _predict_p1(clf, X) for m_name, clf in self.models.items()}

    def route(self, prompt: str, embedding, quality_target: float = None) -> RouteDecision:
        start = time.time()
        qt = quality_target if quality_target is not None else self.quality_target
        p_hat = self._p_hat(prompt, embedding)

        chosen = None
        for m in self.registry.cheapest_to_most_expensive():
            if p_hat.get(m.name, 0.0) >= qt:
                chosen = m.name
                break
        if chosen is None:
            chosen = max(p_hat, key=p_hat.get) if p_hat else self.registry.enabled_models()[0].name

        return RouteDecision(
            chosen, p_hat, {}, "threshold",
            confidence=p_hat.get(chosen, 0.0),
            routing_ms=(time.time() - start) * 1000,
        )


class HandcraftedRouter(_BaseLearnedRouter):
    """(a) handcrafted features only — "the proposal as written"."""
    name = "hgb_handcrafted"

    def _clf_factory(self):
        return HistGradientBoostingClassifier(max_depth=4, random_state=0)

    def _featurize(self, prompt, embedding):
        return extract_features(prompt)


class HandcraftedPlusEmbeddingRouter(_BaseLearnedRouter):
    """(b) handcrafted features plus the embedding."""
    name = "hgb_handcrafted+emb"

    def _clf_factory(self):
        return HistGradientBoostingClassifier(max_depth=4, random_state=0)

    def _featurize(self, prompt, embedding):
        return np.concatenate([extract_features(prompt), np.asarray(embedding, dtype="float32")])


class LogRegEmbeddingRouter(_BaseLearnedRouter):
    """Embedding only, linear model — the simplest learned baseline."""
    name = "logreg_emb"

    def _clf_factory(self):
        return LogisticRegression(max_iter=2000)

    def _featurize(self, prompt, embedding):
        return np.asarray(embedding, dtype="float32")
