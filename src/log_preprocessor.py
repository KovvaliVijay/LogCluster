"""
Log Preprocessor Module for LogCluster.

Performs cleaning, variable-value normalization (IPs, numbers, timestamps, UUIDs, hex),
and tokenization so that variable noise does not fragment log pattern discovery.
"""

import re
from typing import List, Tuple
import pandas as pd


class LogPreprocessor:
    """Preprocesses raw log strings by substituting volatile variables with generalized tokens."""

    # Default regex replacement patterns (applied sequentially)
    DEFAULT_NORMALIZATION_RULES: List[Tuple[re.Pattern, str]] = [
        # ISO / Common Timestamps: e.g. 2026-10-20 10:00:01, 2026-10-20T10:00:01Z, Oct 20 10:00:01
        (re.compile(r"^\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?\s*"), ""),
        (re.compile(r"^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s*"), ""),
        # IPv4 addresses
        (re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?::\d+)?\b"), "<IP>"),
        # UUIDs / GUIDs
        (re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"), "<UUID>"),
        # Hexadecimal addresses/hashes (e.g. 0x7fff, 3a5f8b91)
        (re.compile(r"\b0x[0-9a-fA-F]+\b"), "<HEX>"),
        (re.compile(r"\b[0-9a-fA-F]{8,}\b"), "<HEX>"),
        # Absolute file system paths (e.g. /var/log/syslog, C:\Users\...)
        (re.compile(r"(?:/[a-zA-Z0-9_\.\-]+){2,}"), "<PATH>"),
        (re.compile(r"[a-zA-Z]:\\(?:[a-zA-Z0-9_\.\-]+\\)+[a-zA-Z0-9_\.\-]+"), "<PATH>"),
        # Isolated numbers (e.g., ports, durations, counts, IDs)
        (re.compile(r"\b\d+\b"), "<NUM>"),
        # Collapse multiple spaces
        (re.compile(r"\s+"), " "),
    ]

    def __init__(self, custom_rules: List[Tuple[str, str]] = None):
        """Initialize preprocessor with optional custom regex substitution rules.

        :param custom_rules: Optional list of (pattern_str, replacement_str) tuples.
        """
        self.rules = []
        if custom_rules:
            for pat, repl in custom_rules:
                self.rules.append((re.compile(pat), repl))
        else:
            self.rules = self.DEFAULT_NORMALIZATION_RULES

    def clean_text(self, text: str) -> str:
        """Apply normalization rules to a single log text line.

        :param text: Raw log string.
        :return: Normalized log string.
        """
        result = str(text).strip()
        for pattern, replacement in self.rules:
            result = pattern.sub(replacement, result)
        return result.strip()

    def transform(self, df: pd.DataFrame, source_column: str = "raw_log") -> pd.DataFrame:
        """Process an entire DataFrame of logs, adding 'cleaned_log'.

        :param df: DataFrame containing log data.
        :param source_column: Column name containing raw log messages.
        :return: DataFrame enriched with 'cleaned_log'.
        """
        if source_column not in df.columns:
            raise KeyError(f"Expected column '{source_column}' not found in DataFrame.")

        result_df = df.copy()
        result_df["cleaned_log"] = result_df[source_column].apply(self.clean_text)
        return result_df
