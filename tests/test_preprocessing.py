"""
Unit tests for LogPreprocessor module.
"""

import pandas as pd
from src.log_preprocessor import LogPreprocessor


def test_ip_normalization():
    preprocessor = LogPreprocessor()
    raw = "Connection accepted from 192.168.1.105:8080"
    cleaned = preprocessor.clean_text(raw)
    assert "<IP>" in cleaned
    assert "192.168.1.105" not in cleaned


def test_timestamp_removal():
    preprocessor = LogPreprocessor()
    raw = "2026-10-20 10:00:01 INFO Database initiated"
    cleaned = preprocessor.clean_text(raw)
    assert not cleaned.startswith("2026-10-20")
    assert "Database initiated" in cleaned


def test_uuid_normalization():
    preprocessor = LogPreprocessor()
    raw = "Session 123e4567-e89b-12d3-a456-426614174000 expired"
    cleaned = preprocessor.clean_text(raw)
    assert "<UUID>" in cleaned
    assert "123e4567" not in cleaned


def test_number_normalization():
    preprocessor = LogPreprocessor()
    raw = "User 101 logged in after 20 seconds"
    cleaned = preprocessor.clean_text(raw)
    assert "<NUM>" in cleaned
    assert "101" not in cleaned
    assert "20" not in cleaned


def test_transform_dataframe():
    preprocessor = LogPreprocessor()
    df = pd.DataFrame({"raw_log": ["User 101 logged in", "User 202 logged in"]})
    transformed = preprocessor.transform(df)
    assert "cleaned_log" in transformed.columns
    assert transformed["cleaned_log"].iloc[0] == "User <NUM> logged in"
    assert transformed["cleaned_log"].iloc[1] == "User <NUM> logged in"
