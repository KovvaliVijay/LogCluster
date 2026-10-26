"""
Unit tests for ResultsExporter module.
"""

import pandas as pd
from src.results_exporter import ResultsExporter


def test_export_to_csv_string():
    df = pd.DataFrame({
        "line_number": [1, 2],
        "raw_log": ["User 101 in", "Error DB"],
        "cleaned_log": ["User <NUM> in", "Error DB"],
        "pattern_id": [0, 1],
        "representative_pattern": ["User <NUM> in", "Error DB"],
        "pattern_count": [1, 1],
        "pattern_percentage": [50.0, 50.0],
        "is_rare": [False, True],
    })

    csv_str = ResultsExporter.export_to_csv_string(df)
    assert "line_number" in csv_str
    assert "representative_pattern" in csv_str
    assert "User <NUM> in" in csv_str
    assert "True" in csv_str


def test_export_to_csv_file(tmp_path):
    output_path = tmp_path / "exported_results.csv"
    df = pd.DataFrame({
        "line_number": [1],
        "raw_log": ["test log"],
        "pattern_id": [0],
        "representative_pattern": ["test log"],
        "pattern_count": [1],
        "pattern_percentage": [100.0],
        "is_rare": [False],
    })

    written = ResultsExporter.export_to_csv_file(df, str(output_path))
    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "test log" in content
