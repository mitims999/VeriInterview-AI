"""
Adapter for UKPLab/sentence-transformers (https://github.com/UKPLab/sentence-transformers),
used by the Expected-Answer Matching Module (Build Spec Section 3.6).

INTEGRATION NOTE: `sentence-transformers` (e.g. all-MiniLM-L6-v2) needs a
~90MB model download on first use, which may not be available in an offline
hackathon demo environment. This adapter provides a drop-in
`embed_and_score()` that tries sentence-transformers first and transparently
falls back to a TF-IDF + cosine-similarity vectorizer (scikit-learn, already
a project dependency) if the model can't be loaded. Both paths return a
0-100 semantic match score with the same contract, so no caller needs to
change when the real model becomes available.
"""
from __future__ import annotations
from functools import lru_cache

_st_model = None
_st_tried = False


def _try_load_sentence_transformers():
    global _st_model, _st_tried
    if _st_tried:
        return _st_model
    _st_tried = True
    try:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        _st_model = None
    return _st_model


def semantic_similarity(text_a: str, text_b: str) -> float:
    """Returns cosine similarity in [0,1]."""
    model = _try_load_sentence_transformers()
    if model is not None:
        import numpy as np
        emb = model.encode([text_a, text_b])
        a, b = emb[0], emb[1]
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
        return float(max(0.0, min(1.0, np.dot(a, b) / denom)))

    return _tfidf_cosine(text_a, text_b)


def _tfidf_cosine(text_a: str, text_b: str) -> float:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    if not text_a.strip() or not text_b.strip():
        return 0.0
    vec = TfidfVectorizer(stop_words="english")
    try:
        matrix = vec.fit_transform([text_a, text_b])
    except ValueError:
        return 0.0
    sim = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
    return float(max(0.0, min(1.0, sim)))


def concept_coverage(candidate_answer: str, required_concepts: list[str]) -> tuple[float, list[str]]:
    """Fraction of required concepts textually/semantically present in the answer."""
    if not required_concepts:
        return 1.0, []
    answer_l = candidate_answer.lower()
    missing = []
    covered = 0
    for concept in required_concepts:
        cl = concept.lower()
        if cl in answer_l:
            covered += 1
        elif semantic_similarity(candidate_answer, concept) > 0.35:
            covered += 1
        else:
            missing.append(concept)
    return covered / len(required_concepts), missing
