"""
Unit tests for RarePatternDetector module.
"""

import pandas as pd
from src.rare_pattern_detector import RarePatternDetector


def test_frequency_calculation():
    detector = RarePatternDetector(threshold_value=15.0, threshold_mode="percentage")

    # 10 logs total: 8 of pattern 0 (80%), 2 of pattern 1 (20%)
    df = pd.DataFrame({
        "pattern_id": [0] * 8 + [1] * 2,
        "representative_pattern": ["Common Pattern"] * 8 + ["Rare Pattern"] * 2,
        "raw_log": ["log common"] * 8 + ["log rare"] * 2,
    })

    summary = detector.summarize_patterns(df)
    assert len(summary) == 2

    # Verify counts & percentages
    common_row = summary[summary["pattern_id"] == 0].iloc[0]
    rare_row = summary[summary["pattern_id"] == 1].iloc[0]

    assert common_row["count"] == 8
    assert common_row["percentage"] == 80.0
    assert not common_row["is_rare"]

    assert rare_row["count"] == 2
    assert rare_row["percentage"] == 20.0
    # Threshold is 15%, so 20% is NOT rare under 15%
    assert not rare_row["is_rare"]


def test_dynamic_threshold_update():
    # Show that threshold can be changed without re-running grouping
    summary = pd.DataFrame({
        "pattern_id": [0, 1],
        "representative_pattern": ["Common", "Rare"],
        "count": [98, 2],
        "percentage": [98.0, 2.0],
        "sample_log": ["log 1", "log 2"],
    })

    # Under 1% threshold, 2% is common
    updated_1 = RarePatternDetector.apply_threshold(summary, threshold_value=1.0, threshold_mode="percentage")
    assert not updated_1[updated_1["pattern_id"] == 1]["is_rare"].iloc[0]

    # Under 5% threshold, 2% is classified as rare
    updated_5 = RarePatternDetector.apply_threshold(summary, threshold_value=5.0, threshold_mode="percentage")
    assert updated_5[updated_5["pattern_id"] == 1]["is_rare"].iloc[0]


def test_count_based_threshold():
    summary = pd.DataFrame({
        "pattern_id": [0, 1],
        "count": [10, 2],
        "percentage": [83.3, 16.7],
    })

    updated = RarePatternDetector.apply_threshold(summary, threshold_value=2, threshold_mode="count")
    assert updated.loc[updated["pattern_id"] == 1, "is_rare"].iloc[0]
    assert not updated.loc[updated["pattern_id"] == 0, "is_rare"].iloc[0]


def test_annotate_records():
    df = pd.DataFrame({
        "raw_log": ["a", "b", "c"],
        "pattern_id": [1, 1, 2],
    })
    summary = pd.DataFrame({
        "pattern_id": [1, 2],
        "count": [2, 1],
        "percentage": [66.67, 33.33],
        "is_rare": [False, True],
    })

    annotated = RarePatternDetector.annotate_records(df, summary)
    assert "is_rare" in annotated.columns
    assert "pattern_percentage" in annotated.columns
    assert not annotated.iloc[0]["is_rare"]
    assert annotated.iloc[2]["is_rare"]
