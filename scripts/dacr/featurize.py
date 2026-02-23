from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def simple_prompt_features(prompts: List[str]) -> np.ndarray:
    feats = []
    for p in prompts:
        s = p or ""
        feats.append(
            [
                len(s),
                s.count("\n"),
                int("```" in s),
                int(bool(re.search(r"\b(how|why|what|when|where|which)\b", s.lower()))),
                int("?" in s),
                int(bool(re.search(r"\b(write|implement|code|python|java|c\+\+)\b", s.lower()))),
                int(bool(re.search(r"\bproof|theorem|lemma|derive\b", s.lower()))),
            ]
        )
    return np.asarray(feats, dtype=np.float32)


@dataclass
class TextEmbedder:
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 64
    device: Optional[str] = None

    def __post_init__(self):
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def encode(self, texts: List[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )


def build_query_features(
    df_queries: pd.DataFrame,
    embedder: TextEmbedder,
    domain_onehot: bool = True,
    domain_map: Optional[Dict[str, int]] = None,
) -> Tuple[np.ndarray, Dict[str, int], pd.DataFrame]:
    q = df_queries.drop_duplicates("qid")[["qid", "prompt", "domain"]].copy().reset_index(drop=True)
    prompts = q["prompt"].astype(str).tolist()

    emb = embedder.encode(prompts)
    meta = simple_prompt_features(prompts)

    X = np.concatenate([emb, meta], axis=1)

    out_domain_map: Dict[str, int] = {}
    if domain_onehot:
        domains = q["domain"].astype(str).tolist()

        # IMPORTANT: fit on train, reuse on test
        if domain_map is None:
            uniq = sorted(set(domains))
            out_domain_map = {d: i for i, d in enumerate(uniq)}
        else:
            out_domain_map = dict(domain_map)

        onehot = np.zeros((len(domains), len(out_domain_map)), dtype=np.float32)
        for i, d in enumerate(domains):
            j = out_domain_map.get(d, None)
            if j is not None:
                onehot[i, j] = 1.0
            # else: unseen domain -> all zeros (OOD)

        X = np.concatenate([X, onehot], axis=1)

    return X, out_domain_map, q
