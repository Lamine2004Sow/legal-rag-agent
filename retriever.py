"""Recherche hybride sur les articles de codes juridiques : FAISS (dense) + BM25, fusion RRF.

Usage :
    python retriever.py "ma question" [-k 5]

Les articles cités explicitement (« article 1240 », « L. 221-18 »…) passent en tête
des résultats, avant la fusion RRF du reste.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from ingest import DATA_DIR, DEFAULT_MODEL, load_embeddings, normalize_num

RRF_K = 60

STOPWORDS = frozenset(
    """le la les de des du un une et ou a au aux en que qui est pour par dans sur se ce il elle je on si
    d l s n qu ne pas son sa ses leur leurs plus peut etre sont avec cette tout toute dont ainsi""".split()
)
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

CODE_NAMES = {"code civil": "Code civil", "code de la consommation": "Code de la consommation"}
_NUM = r"(?:[LRDA]\.?\s*\*?)?\d+(?:-\d+)*"
# « article 1240 », « art. L. 221-18 », « articles 1240 et 1241 du code civil »
_ARTICLE_RE = re.compile(
    rf"\b(?:articles?|art\.?)\s+(?P<nums>{_NUM}(?:\s*(?:,|et|ou)\s*{_NUM})*)(?P<tail>.{{0,60}})",
    re.IGNORECASE,
)
# Forme sans mot « article » : « L221-18 », « R. 314-20 ». Majuscule et tiret obligatoires,
# pour ne pas prendre « 30 jours » ni « 2024 ».
_BARE_RE = re.compile(r"\b[LRD]\.?\s*\*?\d+(?:-\d+)+\b")
_CODE_RE = re.compile(r"\s*,?\s*(?:du|de|of)\s+(code civil|code de la consommation)\b", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Minuscules, sans accents, mots vides retirés. « l221-18 » reste un seul token."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return [t for t in _TOKEN_RE.findall(text) if t not in STOPWORDS]


def find_article_refs(query: str) -> list[tuple[str | None, str]]:
    """Références explicites de la question -> [(code détecté ou None, num normalisé)]."""
    refs: list[tuple[str | None, str]] = []
    for m in _ARTICLE_RE.finditer(query):
        code_m = _CODE_RE.match(m.group("tail"))
        code = CODE_NAMES[code_m.group(1).lower()] if code_m else None
        for raw in re.findall(_NUM, m.group("nums")):
            ref = (code, normalize_num(raw.upper()))
            if ref not in refs:
                refs.append(ref)
    for m in _BARE_RE.finditer(query):
        ref = (None, normalize_num(m.group(0).upper()))
        if ref not in refs:
            refs.append(ref)
    return refs


class HybridRetriever:
    def __init__(self, data_dir: Path = DATA_DIR, model_name: str = DEFAULT_MODEL) -> None:
        self.embeddings = load_embeddings(model_name)
        self.index = FAISS.load_local(
            str(data_dir / "faiss_index"),
            self.embeddings,
            allow_dangerous_deserialization=True,
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        )
        self.catalog = pd.read_parquet(data_dir / "articles.parquet")

        # chunk_id = position FAISS (0..N-1) ; BM25 est construit dans le même ordre.
        store = self.index.docstore._dict
        self.chunks: dict[int, Document] = {
            i: store[self.index.index_to_docstore_id[i]] for i in range(self.index.index.ntotal)
        }
        self.bm25 = BM25Okapi([tokenize(d.page_content) for d in self.chunks.values()])

        self._by_article: dict[tuple[str, str], list[int]] = defaultdict(list)
        for i, doc in self.chunks.items():
            self._by_article[(doc.metadata["code"], doc.metadata["num"])].append(i)

    def get_article(self, code: str | None, num: str) -> dict | None:
        rows = self.catalog[self.catalog["num"] == normalize_num(num)]
        if code is not None:
            rows = rows[rows["code"] == code]
        return rows.iloc[0].to_dict() if not rows.empty else None

    def search(self, query: str, k: int = 5, n_candidates: int = 30) -> list[Document]:
        # Dense : même recherche que similarity_search_with_score, mais on garde les positions FAISS.
        vec = np.array([self.embeddings.embed_query(query)], dtype="float32")
        d_scores, d_ids = self.index.index.search(vec, n_candidates)
        dense = {int(i): float(s) for s, i in zip(d_scores[0], d_ids[0]) if i != -1}

        # BM25 : top n_candidates ayant au moins un terme en commun.
        bm = self.bm25.get_scores(tokenize(query))
        bm25 = {int(i): float(bm[i]) for i in np.argsort(-bm)[:n_candidates] if bm[i] > 0}

        # Fusion RRF.
        rrf: dict[int, float] = defaultdict(float)
        for ranking in (dense, bm25):
            for rank, i in enumerate(ranking, 1):
                rrf[i] += 1 / (RRF_K + rank)

        # Articles cités explicitement, en tête ; puis le reste par RRF décroissant.
        explicit: list[int] = []
        for code, num in find_article_refs(query):
            art = self.get_article(code, num)
            if art is not None:
                explicit += self._by_article.get((art["code"], art["num"]), [])
        explicit = list(dict.fromkeys(explicit))
        rest = sorted((i for i in rrf if i not in explicit), key=rrf.get, reverse=True)
        ordered = [(i, "exact") for i in explicit] + [(i, "hybrid") for i in rest]

        results = []
        for i, match in ordered[:k]:
            doc = self.chunks[i]
            meta = {
                **doc.metadata,
                "dense_score": dense.get(i),
                "bm25_score": bm25.get(i),
                "rrf_score": rrf.get(i, 0.0),
                "match": match,
            }
            results.append(Document(page_content=doc.page_content, metadata=meta))
        return results


def _fmt(score: float | None) -> str:
    return "-" if score is None else f"{score:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    retriever = HybridRetriever()
    for rank, doc in enumerate(retriever.search(args.query, k=args.k), 1):
        m = doc.metadata
        print(
            f"{rank}. {m['code']} art. {m['num']} [{m['match']}] "
            f"dense={_fmt(m['dense_score'])} bm25={_fmt(m['bm25_score'])} rrf={m['rrf_score']:.4f}"
        )
        print(f"   {doc.page_content[:120].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
