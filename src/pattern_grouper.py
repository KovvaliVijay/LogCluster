"""
Pattern Grouper Module for LogCluster.

Groups structurally similar log messages into recurring patterns using unsupervised
learning and template extraction techniques (Drain algorithm or TF-IDF + Clustering).
Assigns Pattern IDs, extracts representative templates, and links to raw messages.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN


class DrainNode:
    """Prefix tree node for the Drain log parsing algorithm."""
    __slots__ = ["children", "cluster_ids"]

    def __init__(self):
        self.children: Dict[str, "DrainNode"] = {}
        self.cluster_ids: List[int] = []


class DrainParser:
    """Lightweight, efficient implementation of the Drain tree-based log parser.

    Groups log messages into templates by token length and token prefixes, replacing
    differing tokens with a wildcard token '<*>'.
    """

    def __init__(self, depth: int = 4, sim_th: float = 0.5, max_children: int = 100):
        self.depth = depth - 2  # internal search depth
        self.sim_th = sim_th
        self.max_children = max_children
        self.root = DrainNode()
        self.clusters: List[Dict] = []  # List of {'id': int, 'tokens': list, 'template': str, 'size': int}

    def _has_variable(self, token: str) -> bool:
        """Check if a token appears to be a variable (contains numbers or wildcard markers)."""
        return any(ch.isdigit() for ch in token) or token.startswith("<") and token.endswith(">")

    def _token_similarity(self, seq1: List[str], seq2: List[str]) -> Tuple[float, int]:
        """Compute token-level similarity between two sequences of identical length."""
        if len(seq1) != len(seq2) or not seq1:
            return 0.0, 0
        matches = 0
        for t1, t2 in zip(seq1, seq2):
            if t1 == "<*>" or t1 == t2:
                matches += 1
        return matches / len(seq1), matches

    def add_log_message(self, message: str) -> int:
        """Parse a single cleaned message, assign or create a cluster, and return its cluster ID."""
        tokens = message.strip().split()
        if not tokens:
            tokens = ["<EMPTY>"]

        log_len = len(tokens)
        len_key = str(log_len)

        # Level 1: Length node
        if len_key not in self.root.children:
            len_node = DrainNode()
            self.root.children[len_key] = len_node
        else:
            len_node = self.root.children[len_key]

        # Level 2..depth: Prefix tokens
        curr_node = len_node
        depth_limit = min(self.depth, log_len)
        for i in range(depth_limit):
            token = tokens[i]
            # If variable, mask key
            key = "<*>" if self._has_variable(token) else token

            if key not in curr_node.children:
                if len(curr_node.children) >= self.max_children:
                    key = "<*>"
                    if key not in curr_node.children:
                        curr_node.children[key] = DrainNode()
                else:
                    curr_node.children[key] = DrainNode()

            curr_node = curr_node.children[key]

        # Find best matching cluster among candidates
        best_cluster = None
        max_sim = -1.0
        for cid in curr_node.cluster_ids:
            cand = self.clusters[cid]
            sim, _ = self._token_similarity(cand["tokens"], tokens)
            if sim > max_sim:
                max_sim = sim
                best_cluster = cand

        if max_sim >= self.sim_th and best_cluster is not None:
            # Update matching cluster template with wildcard for differing tokens
            updated_tokens = []
            for t_old, t_new in zip(best_cluster["tokens"], tokens):
                if t_old == t_new:
                    updated_tokens.append(t_old)
                else:
                    updated_tokens.append("<*>")
            best_cluster["tokens"] = updated_tokens
            best_cluster["template"] = " ".join(updated_tokens)
            best_cluster["size"] += 1
            return best_cluster["id"]
        else:
            # Create a new cluster
            new_id = len(self.clusters)
            new_cluster = {
                "id": new_id,
                "tokens": list(tokens),
                "template": " ".join(tokens),
                "size": 1,
            }
            self.clusters.append(new_cluster)
            curr_node.cluster_ids.append(new_id)
            return new_id


class PatternGrouper:
    """Unsupervised log pattern discovery and grouping engine."""

    def __init__(
        self,
        method: str = "drain",
        sim_threshold: float = 0.5,
        n_clusters: int = 5,
        dbscan_eps: float = 0.3,
        dbscan_min_samples: int = 2,
    ):
        """Initialize pattern grouper.

        :param method: Grouping algorithm ('drain', 'kmeans', or 'dbscan').
        :param sim_threshold: Similarity threshold for Drain parser (0.1 - 1.0).
        :param n_clusters: Number of clusters for KMeans.
        :param dbscan_eps: Neighborhood distance epsilon for DBSCAN.
        :param dbscan_min_samples: Minimum cluster size for DBSCAN.
        """
        self.method = method.lower()
        self.sim_threshold = sim_threshold
        self.n_clusters = n_clusters
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples

    def fit_predict(self, df: pd.DataFrame, text_column: str = "cleaned_log") -> pd.DataFrame:
        """Discover patterns in the log DataFrame and annotate each record with its pattern.

        Adds:
            - 'pattern_id': Integer cluster/pattern identifier (or 'NOISE' for DBSCAN outliers).
            - 'representative_pattern': Human-readable pattern template string.

        :param df: DataFrame containing cleaned log messages.
        :param text_column: Column name to use for pattern discovery.
        :return: Enriched DataFrame.
        """
        if text_column not in df.columns:
            raise KeyError(f"Expected column '{text_column}' not found in DataFrame.")

        result_df = df.copy()
        texts = result_df[text_column].astype(str).tolist()

        if not texts:
            result_df["pattern_id"] = []
            result_df["representative_pattern"] = []
            return result_df

        if self.method == "drain":
            return self._group_by_drain(result_df, texts)
        elif self.method == "kmeans":
            return self._group_by_kmeans(result_df, texts)
        elif self.method == "dbscan":
            return self._group_by_dbscan(result_df, texts)
        else:
            raise ValueError(f"Unsupported pattern grouping method: '{self.method}'. Use 'drain', 'kmeans', or 'dbscan'.")

    def _group_by_drain(self, df: pd.DataFrame, texts: List[str]) -> pd.DataFrame:
        """Group logs using tree-based Drain template extraction."""
        parser = DrainParser(sim_th=self.sim_threshold)
        pattern_ids = []
        for line in texts:
            pid = parser.add_log_message(line)
            pattern_ids.append(pid)

        templates_by_id = {c["id"]: c["template"] for c in parser.clusters}

        df["pattern_id"] = pattern_ids
        df["representative_pattern"] = [templates_by_id[pid] for pid in pattern_ids]
        return df

    def _group_by_kmeans(self, df: pd.DataFrame, texts: List[str]) -> pd.DataFrame:
        """Group logs using TF-IDF vectorization and KMeans clustering."""
        vectorizer = TfidfVectorizer(max_features=1000, stop_words="english", token_pattern=r"(?u)\b\w+\b")
        X = vectorizer.fit_transform(texts)

        k = min(self.n_clusters, len(texts))
        if k < 1:
            k = 1

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        df["pattern_id"] = labels

        # Determine representative template for each cluster (most frequent cleaned message in cluster)
        rep_patterns = {}
        for c in range(k):
            cluster_texts = df.loc[df["pattern_id"] == c, "cleaned_log"]
            if not cluster_texts.empty:
                rep_patterns[c] = cluster_texts.mode().iloc[0]
            else:
                rep_patterns[c] = f"Cluster {c}"

        df["representative_pattern"] = df["pattern_id"].map(rep_patterns)
        return df

    def _group_by_dbscan(self, df: pd.DataFrame, texts: List[str]) -> pd.DataFrame:
        """Group logs using TF-IDF vectorization and DBSCAN density clustering.

        Explicitly handles noise points (label -1) as distinct outlier points.
        """
        vectorizer = TfidfVectorizer(max_features=1000, stop_words="english", token_pattern=r"(?u)\b\w+\b")
        X = vectorizer.fit_transform(texts)

        dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples, metric="cosine")
        labels = dbscan.fit_predict(X)

        # Scikit-learn assigns -1 to noise/outliers. Explicitly handle them:
        formatted_labels = []
        for lbl in labels:
            if lbl == -1:
                formatted_labels.append(-1)  # Distinct noise identifier
            else:
                formatted_labels.append(lbl)

        df["pattern_id"] = formatted_labels

        rep_patterns = {}
        unique_labels = set(formatted_labels)
        for lbl in unique_labels:
            if lbl == -1:
                rep_patterns[-1] = "<DBSCAN_NOISE_POINT / UNGROUPED_OUTLIER>"
            else:
                cluster_texts = df.loc[df["pattern_id"] == lbl, "cleaned_log"]
                if not cluster_texts.empty:
                    rep_patterns[lbl] = cluster_texts.mode().iloc[0]
                else:
                    rep_patterns[lbl] = f"Cluster {lbl}"

        df["representative_pattern"] = df["pattern_id"].map(rep_patterns)
        return df
