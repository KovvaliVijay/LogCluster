"""
Unit tests for PatternGrouper module.
"""

import pandas as pd
from src.pattern_grouper import PatternGrouper


def test_drain_pattern_grouping():
    grouper = PatternGrouper(method="drain", sim_threshold=0.5)

    logs = [
        "User <NUM> logged in",
        "User <NUM> logged in",
        "User <NUM> logged in",
        "Database connection failed",
        "Database connection failed",
        "Request timed out after <NUM> seconds",
    ]
    df = pd.DataFrame({"cleaned_log": logs, "raw_log": logs})
    result = grouper.fit_predict(df)

    assert "pattern_id" in result.columns
    assert "representative_pattern" in result.columns

    # The 3 user logs should share the exact same pattern_id
    user_pids = result.iloc[0:3]["pattern_id"].unique()
    assert len(user_pids) == 1

    # Database logs should share a pattern_id
    db_pids = result.iloc[3:5]["pattern_id"].unique()
    assert len(db_pids) == 1

    # Total distinct patterns should be 3
    assert len(result["pattern_id"].unique()) == 3


def test_kmeans_pattern_grouping():
    grouper = PatternGrouper(method="kmeans", n_clusters=2)

    logs = [
        "User logged in successfully",
        "User logged in successfully",
        "Fatal kernel panic occurred",
        "Fatal kernel panic occurred",
    ]
    df = pd.DataFrame({"cleaned_log": logs, "raw_log": logs})
    result = grouper.fit_predict(df)

    assert len(result["pattern_id"].unique()) == 2
    assert "representative_pattern" in result.columns


def test_dbscan_noise_handling():
    # DBSCAN with high min_samples to verify noise handling
    grouper = PatternGrouper(method="dbscan", dbscan_eps=0.1, dbscan_min_samples=3)

    logs = [
        "Repeated standard log line",
        "Repeated standard log line",
        "Repeated standard log line",
        "Completely unique random outlier line xyz",
    ]
    df = pd.DataFrame({"cleaned_log": logs, "raw_log": logs})
    result = grouper.fit_predict(df)

    assert "pattern_id" in result.columns
    # Check that noise is explicitly identified and not discarded
    noise_rows = result[result["pattern_id"] == -1]
    assert len(noise_rows) >= 1
    assert "NOISE" in noise_rows["representative_pattern"].iloc[0] or "OUTLIER" in noise_rows["representative_pattern"].iloc[0]
