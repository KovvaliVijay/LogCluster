"""
Unit tests for Icons module.
"""

from src.icons import get_icon, get_logo_brand, ICON_MAP


def test_icon_mapping():
    assert "dashboard" in ICON_MAP
    assert "sun" in ICON_MAP
    assert "moon" in ICON_MAP
    assert "patterns" in ICON_MAP
    assert "rare" in ICON_MAP
    assert "search" in ICON_MAP
    assert "export" in ICON_MAP


def test_get_icon():
    icon = get_icon("dashboard", size=20, className="test-class", aria_label="Dashboard Icon")
    assert "bi-speedometer2" in icon.className
    assert "test-class" in icon.className
    assert icon.style["fontSize"] == "20px"
    assert icon.role == "img"


def test_get_logo_brand():
    logo = get_logo_brand(size=36, className="brand-logo")
    assert "/assets/icons/logo.svg" in logo.src
    assert logo.style["width"] == "36px"
    assert logo.alt == "LogCluster Logo"
