"""Ingestion : codes juridiques (Tricoteuses/LEGI) -> catalogue d'articles + index FAISS.

Usage :
    python ingest.py
    python ingest.py --codes "Code civil" --model intfloat/multilingual-e5-base

Produit dans data/ :
    articles.parquet  catalogue dédoublonné (sert à vérifier que les citations existent)
    faiss_index/      index vectoriel des chunks
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

DATASET_REPO = "Tricoteuses/LEGI"
# Révision figée : le jeu de test et la non-régression supposent un corpus stable.
DATASET_REVISION = "48fe0f8e487655d7012d799a88eeadb42f0b021d"
DATASET_FILE = "data/LEGI.parquet"

DEFAULT_CODES = ("Code civil", "Code de la consommation")
DEFAULT_MODEL = "intfloat/multilingual-e5-small"
DATA_DIR = Path("data")

# e5 tronque à 512 tokens ; ~300 mots français restent en dessous.
MAX_WORDS = 300

LEGIFRANCE_ARTICLE_URL = "https://www.legifrance.gouv.fr/codes/article_lc/{id}"

_LEVEL_RE = re.compile(
    r"(?=\b(?:Partie|Livre|Sous-titre|Titre|Chapitre|Sous-section|Section|Paragraphe|Sous-paragraphe)\b"
    r"[^:]{0,40}:)"
)
# Coupe après . ; : devant une majuscule ou une énumération (« 1° », « a) »),
# sans casser les références « art. L. 121-1 ». Le texte omet souvent l'espace (« :1° »).
_SENTENCE_RE = re.compile(
    r"(?<=[.;:])(?<!\bart\.)(?<!\b[LRDA]\.)\s*(?=\d+°|[a-z]\)\s|[A-ZÀÂÉÈÊÎÔÛÇ«])"
)


class E5Embeddings(HuggingFaceEmbeddings):
    """Les modèles e5 attendent les préfixes « passage: » et « query: »."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents([f"passage: {t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(f"query: {text}")


def load_embeddings(model_name: str = DEFAULT_MODEL) -> E5Embeddings:
    return E5Embeddings(model_name=model_name, encode_kwargs={"normalize_embeddings": True})


def normalize_num(num: str) -> str:
    """« L. 121-1 », « R*121-1 », «  L121-1 » -> « L121-1 ».

    Utilisé à l'ingestion et pour résoudre les citations produites par le LLM.
    """
    num = num.strip().replace("*", "")
    if num.lower() in ("liminaire", "article liminaire"):
        return "Article liminaire"
    num = re.sub(r"^([LRDA])\.?\s*", r"\1", num)
    return re.sub(r"\s+", " ", num)


def clean_titre(titre: str) -> str:
    """Retire les niveaux répétés : « Titre III : A Titre III : B » -> « Titre III : A »."""
    parts = [p.strip() for p in _LEVEL_RE.split(titre) if p.strip()]
    kept, seen_labels = [], set()
    for part in parts:
        label = part.split(":", 1)[0].strip()
        if label in seen_labels:
            continue
        seen_labels.add(label)
        kept.append(part)
    return " > ".join(kept)


def load_articles(codes: tuple[str, ...]) -> pd.DataFrame:
    path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=DATASET_FILE,
        repo_type="dataset",
        revision=DATASET_REVISION,
    )
    columns = ["id", "source", "num", "titre", "contenu", "nota", "date_debut", "etat"]
    df = pd.read_parquet(path, columns=columns)
    missing = set(codes) - set(df["source"].unique())
    if missing:
        raise ValueError(f"Codes absents du dataset : {sorted(missing)}")
    return df[df["source"].isin(codes) & (df["etat"] == "VIGUEUR")]


def prepare_articles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"source": "code"}).copy()
    df["num"] = df["num"].map(normalize_num)
    df["titre"] = df["titre"].map(clean_titre)
    df["contenu"] = df["contenu"].str.strip()
    df = df[df["contenu"] != ""]

    # Le dataset garde parfois plusieurs versions « VIGUEUR » d'un même article
    # (ex. Code civil art. 1842 : 1978 et 2024). On garde la plus récente.
    df = df.sort_values(["date_debut", "id"]).drop_duplicates(["code", "num"], keep="last")

    df["url"] = df["id"].map(lambda i: LEGIFRANCE_ARTICLE_URL.format(id=i))
    return df.drop(columns="etat").sort_values(["code", "num"]).reset_index(drop=True)


def split_long(text: str, max_words: int = MAX_WORDS) -> list[str]:
    """Découpe par phrases en morceaux de ≤ max_words, avec une phrase de chevauchement."""
    if len(text.split()) <= max_words:
        return [text]
    sentences = []
    for sentence in _SENTENCE_RE.split(text):
        words = sentence.split()
        # Phrase trop longue à elle seule : fenêtres de mots.
        sentences += [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]

    chunks, current = [], []
    for sentence in sentences:
        if current and len(" ".join(current + [sentence]).split()) > max_words:
            chunks.append(" ".join(current))
            overlap = current[-1]
            current = [overlap] if len(f"{overlap} {sentence}".split()) <= max_words else []
        current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


def build_documents(articles: pd.DataFrame, max_words: int = MAX_WORDS) -> list[Document]:
    docs = []
    for art in articles.itertuples(index=False):
        header = f"{art.code}, art. {art.num}\n{art.titre}"
        parts = split_long(art.contenu, max_words)
        for i, part in enumerate(parts):
            docs.append(
                Document(
                    page_content=f"{header}\n\n{part}",
                    metadata={
                        "article_id": art.id,
                        "code": art.code,
                        "num": art.num,
                        "url": art.url,
                        "chunk": i,
                        "n_chunks": len(parts),
                    },
                )
            )
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--codes", nargs="+", default=list(DEFAULT_CODES))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DATA_DIR)
    parser.add_argument("--max-words", type=int, default=MAX_WORDS)
    args = parser.parse_args()

    raw = load_articles(tuple(args.codes))
    articles = prepare_articles(raw)
    print(f"{len(raw)} articles chargés, {len(articles)} après dédoublonnage")
    print(articles.groupby("code").size().to_string())

    args.out.mkdir(parents=True, exist_ok=True)
    articles.to_parquet(args.out / "articles.parquet", index=False)

    docs = build_documents(articles, args.max_words)
    print(f"{len(docs)} chunks, embedding avec {args.model}…")
    index = FAISS.from_documents(
        docs,
        load_embeddings(args.model),
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )
    index.save_local(str(args.out / "faiss_index"))
    print(f"Index écrit dans {args.out / 'faiss_index'}")


if __name__ == "__main__":
    main()
