"""
Unit tests for LogLoader module.
"""

import io
import os
import pandas as pd
from src.log_loader import LogLoader


def test_load_text_file(tmp_path):
    log_file = tmp_path / "test.log"
    log_file.write_text("line 1\nline 2\nline 3\n", encoding="utf-8")

    df = LogLoader.load(str(log_file))
    assert len(df) == 3
    assert list(df["line_number"]) == [1, 2, 3]
    assert df["raw_log"].iloc[0] == "line 1"


def test_load_empty_file(tmp_path):
    empty_file = tmp_path / "empty.log"
    empty_file.write_text("", encoding="utf-8")

    threw = False
    try:
        LogLoader.load(str(empty_file))
    except ValueError:
        threw = True
    assert threw, "Expected ValueError when loading empty file"


def test_load_nonexistent_file():
    threw = False
    try:
        LogLoader.load("non_existent_path_xyz123.log")
    except FileNotFoundError:
        threw = True
    assert threw, "Expected FileNotFoundError when file doesn't exist"


def test_load_csv_file(tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("timestamp,level,message\n2026-10-20,INFO,User connected\n2026-10-20,ERROR,Failed\n", encoding="utf-8")

    df = LogLoader.load(str(csv_file), message_column="message")
    assert len(df) == 2
    assert df["raw_log"].iloc[0] == "User connected"
    assert df["raw_log"].iloc[1] == "Failed"


def test_load_csv_autodetect_column(tmp_path):
    csv_file = tmp_path / "auto.csv"
    csv_file.write_text("time,service,msg\n10:00,auth,Login ok\n10:01,auth,Login fail\n", encoding="utf-8")

    df = LogLoader.load(str(csv_file))
    assert len(df) == 2
    assert "Login ok" in df["raw_log"].iloc[0]


def test_load_from_bytesio():
    data = io.BytesIO(b"log entry alpha\nlog entry beta\n")
    df = LogLoader.load(data, filename="sample.txt")
    assert len(df) == 2
    assert df["raw_log"].iloc[0] == "log entry alpha"


def test_unsupported_format(tmp_path):
    bad_file = tmp_path / "archive.zip"
    bad_file.write_text("dummy", encoding="utf-8")

    threw = False
    try:
        LogLoader.load(str(bad_file))
    except ValueError:
        threw = True
    assert threw, "Expected ValueError for unsupported format"
