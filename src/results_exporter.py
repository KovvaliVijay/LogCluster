"""
Results Exporter Module for LogCluster.

Handles formatting and exporting analysis results (raw logs, discovered patterns,
frequencies, and rarity classifications) to CSV files and in-memory streams.
"""

import io
from typing import Optional, Union
import pandas as pd


class ResultsExporter:
    """Exports log clustering and pattern discovery results to standard CSV format."""

    EXPORT_COLUMNS = [
        "line_number",
        "raw_log",
        "cleaned_log",
        "pattern_id",
        "representative_pattern",
        "pattern_count",
        "pattern_percentage",
        "is_rare",
    ]

    @classmethod
    def prepare_export_df(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Format and reorder columns for export.

        :param df: Analysis DataFrame.
        :return: Cleaned DataFrame with standard export columns.
        """
        available_cols = [c for c in cls.EXPORT_COLUMNS if c in df.columns]
        export_df = df[available_cols].copy()

        # Format percentage nicely
        if "pattern_percentage" in export_df.columns:
            export_df["pattern_percentage"] = export_df["pattern_percentage"].round(2)

        return export_df

    @classmethod
    def export_to_csv_string(cls, df: pd.DataFrame) -> str:
        """Export analysis results to a CSV string.

        :param df: Enriched analysis DataFrame.
        :return: CSV-formatted string.
        """
        export_df = cls.prepare_export_df(df)
        return export_df.to_csv(index=False)

    @classmethod
    def export_to_csv_file(cls, df: pd.DataFrame, filepath: str) -> str:
        """Save analysis results directly to a local CSV file.

        :param df: Enriched analysis DataFrame.
        :param filepath: Target destination path.
        :return: The written filepath.
        """
        export_df = cls.prepare_export_df(df)
        export_df.to_csv(filepath, index=False)
        return filepath
