"""
Rare Pattern Detector Module for LogCluster.

Performs occurrence-frequency statistical analysis on discovered log patterns.
Computes occurrence counts, dataset percentages, and identifies infrequent (rare)
patterns based on a configurable threshold without requiring separate anomaly models.
"""

from typing import Dict, Optional, Tuple
import pandas as pd


class RarePatternDetector:
    """Frequency-based rare pattern detection engine.

    A pattern is classified as 'rare' when its occurrence percentage (or absolute count)
    falls at or below the configured threshold. Infrequent patterns represent statistical
    outliers or uncommon operational events, rather than confirmed system failures or threats.
    """

    def __init__(self, threshold_value: float = 5.0, threshold_mode: str = "percentage"):
        """Initialize detector.

        :param threshold_value: Frequency threshold cutoff (e.g., 5.0 for <= 5.0% of all logs).
        :param threshold_mode: 'percentage' (default) or 'count'.
        """
        self.threshold_value = threshold_value
        self.threshold_mode = threshold_mode

    def summarize_patterns(
        self,
        df: pd.DataFrame,
        pattern_id_col: str = "pattern_id",
        rep_col: str = "representative_pattern",
        raw_col: str = "raw_log",
    ) -> pd.DataFrame:
        """Calculate frequency statistics for each pattern in the dataset.

        Returns a summary DataFrame with:
            - pattern_id
            - representative_pattern
            - count (number of occurrences)
            - percentage (occurrence frequency in %)
            - is_rare (boolean classification)
            - sample_log (an example raw logline)

        :param df: DataFrame containing grouped log records.
        :param pattern_id_col: Column containing pattern identifiers.
        :param rep_col: Column containing representative pattern strings.
        :param raw_col: Column containing raw original log strings.
        :return: Summary DataFrame sorted by occurrence count descending.
        """
        if df.empty or pattern_id_col not in df.columns:
            return pd.DataFrame(columns=[
                "pattern_id", "representative_pattern", "count", "percentage", "is_rare", "sample_log"
            ])

        total_logs = len(df)

        # Aggregate counts and representative patterns
        grouped = df.groupby(pattern_id_col)
        counts = grouped.size()
        rep_patterns = grouped[rep_col].first()
        sample_logs = grouped[raw_col].first()

        summary_df = pd.DataFrame({
            "pattern_id": counts.index,
            "representative_pattern": rep_patterns.values,
            "count": counts.values,
            "percentage": (counts.values / total_logs) * 100.0,
            "sample_log": sample_logs.values,
        })

        # Apply rarity classification
        summary_df = self.apply_threshold(summary_df, self.threshold_value, self.threshold_mode)

        # Sort common patterns first
        summary_df = summary_df.sort_values(by="count", ascending=False).reset_index(drop=True)
        return summary_df

    @staticmethod
    def apply_threshold(
        summary_df: pd.DataFrame,
        threshold_value: float = 5.0,
        threshold_mode: str = "percentage",
    ) -> pd.DataFrame:
        """Re-evaluate rarity classification dynamically without re-running grouping algorithms.

        :param summary_df: Existing pattern summary DataFrame.
        :param threshold_value: New threshold value.
        :param threshold_mode: 'percentage' or 'count'.
        :return: Updated summary DataFrame with recalculated 'is_rare' column.
        """
        updated_df = summary_df.copy()
        if updated_df.empty:
            updated_df["is_rare"] = []
            return updated_df

        if threshold_mode == "count":
            updated_df["is_rare"] = updated_df["count"] <= threshold_value
        else:  # percentage mode
            updated_df["is_rare"] = updated_df["percentage"] <= threshold_value

        return updated_df

    @staticmethod
    def annotate_records(df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
        """Annotate individual log records with their pattern frequency and rarity status.

        :param df: The record-level log DataFrame.
        :param summary_df: The pattern-level summary DataFrame.
        :return: Merged DataFrame containing 'pattern_count', 'pattern_percentage', and 'is_rare'.
        """
        if df.empty or summary_df.empty:
            df["pattern_count"] = 0
            df["pattern_percentage"] = 0.0
            df["is_rare"] = False
            return df

        cols_to_merge = summary_df[["pattern_id", "count", "percentage", "is_rare"]].rename(
            columns={"count": "pattern_count", "percentage": "pattern_percentage"}
        )

        merged = df.merge(cols_to_merge, on="pattern_id", how="left")
        merged["is_rare"] = merged["is_rare"].fillna(False)
        return merged
