"""
Log Loader Module for LogCluster.

Handles ingestion of heterogeneous log files in multiple formats (.log, .txt, .csv).
Preserves raw loglines, extracts message bodies, and handles malformed or empty inputs gracefully.
"""

import io
import os
from typing import Optional, Tuple, Union
import pandas as pd


class LogLoader:
    """Universal log ingestion component supporting .log, .txt, and .csv formats."""

    SUPPORTED_EXTENSIONS = {".log", ".txt", ".csv"}

    @staticmethod
    def inspect_csv_columns(filepath_or_buffer: Union[str, io.StringIO, io.BytesIO]) -> list:
        """Inspect and return available columns in a CSV log file.

        :param filepath_or_buffer: File path or stream buffer.
        :return: List of column names.
        """
        try:
            df = pd.read_csv(filepath_or_buffer, nrows=5)
            return list(df.columns)
        except Exception as exc:
            raise ValueError(f"Unable to read CSV columns: {exc}")

    @classmethod
    def load(
        cls,
        source: Union[str, io.StringIO, io.BytesIO],
        filename: Optional[str] = None,
        message_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load logs from a file path or in-memory stream into a unified DataFrame.

        The resulting DataFrame contains:
            - 'raw_log': Original verbatim log line.
            - 'line_number': 1-indexed line position.
            - Any additional structured columns (if loaded from CSV).

        :param source: File path string or file-like buffer.
        :param filename: Optional filename used to infer format when source is a buffer.
        :param message_column: For CSV files, the column holding the log text.
        :return: Cleaned pandas DataFrame of ingested logs.
        """
        # Determine format / extension
        ext = ""
        if isinstance(source, str):
            if not os.path.exists(source):
                raise FileNotFoundError(f"Log file not found: {source}")
            ext = os.path.splitext(source)[1].lower()
            if not ext and filename:
                ext = os.path.splitext(filename)[1].lower()
        elif filename:
            ext = os.path.splitext(filename)[1].lower()

        if ext not in cls.SUPPORTED_EXTENSIONS and ext != "":
            raise ValueError(
                f"Unsupported file format '{ext}'. Supported formats: {', '.join(cls.SUPPORTED_EXTENSIONS)}"
            )

        if ext == ".csv":
            return cls._load_csv(source, message_column)
        else:
            return cls._load_text(source)

    @classmethod
    def _load_text(cls, source: Union[str, io.StringIO, io.BytesIO]) -> pd.DataFrame:
        """Load unstructured text/log files line by line."""
        lines = []
        if isinstance(source, str):
            with open(source, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        elif isinstance(source, io.BytesIO):
            content = source.getvalue().decode("utf-8", errors="replace")
            lines = content.splitlines()
        elif isinstance(source, io.StringIO):
            lines = source.getvalue().splitlines()
        else:
            lines = [line.decode("utf-8", errors="replace") if isinstance(line, bytes) else str(line) for line in source]

        cleaned_lines = [line.rstrip("\r\n") for line in lines if line.strip()]

        if not cleaned_lines:
            raise ValueError("The provided log file is empty or contains only whitespace.")

        df = pd.DataFrame({
            "raw_log": cleaned_lines,
            "line_number": list(range(1, len(cleaned_lines) + 1)),
        })
        return df

    @classmethod
    def _load_csv(
        cls,
        source: Union[str, io.StringIO, io.BytesIO],
        message_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load structured CSV log files and extract or identify the message column."""
        try:
            df = pd.read_csv(source, on_bad_lines="skip")
        except Exception as exc:
            raise ValueError(f"Malformed or unreadable CSV file: {exc}")

        if df.empty:
            raise ValueError("The provided CSV file contains no data rows.")

        # If message_column specified, verify it exists
        if message_column and message_column in df.columns:
            selected_col = message_column
        else:
            # Auto-detect likely log message column name
            candidates = ["message", "log", "logline", "content", "msg", "body", "Details", "text", "description"]
            selected_col = None
            for cand in candidates:
                for col in df.columns:
                    if col.lower() == cand.lower():
                        selected_col = col
                        break
                if selected_col:
                    break

            # Fallback to the text column with the longest average string length
            if not selected_col:
                str_cols = [c for c in df.columns if df[c].dtype == object]
                if str_cols:
                    selected_col = max(str_cols, key=lambda c: df[c].astype(str).str.len().mean())
                else:
                    selected_col = df.columns[-1]

        df["raw_log"] = df[selected_col].astype(str).str.strip()
        df = df[df["raw_log"].str.len() > 0].reset_index(drop=True)

        if df.empty:
            raise ValueError(f"Selected CSV column '{selected_col}' contains no valid log messages.")

        df["line_number"] = df.index + 1
        return df
