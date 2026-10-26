"""
Icon System Module for LogCluster.

Provides consistent, accessible vector UI icons across Light and Dark themes.
Maps semantic icon names to Bootstrap vector glyphs and custom SVG assets.
"""

from typing import Optional
from dash import html


ICON_MAP = {
    # Navigation & Core Tabs
    "dashboard": "speedometer2",
    "upload": "cloud-arrow-up",
    "patterns": "diagram-3",
    "rare": "exclamation-diamond",
    "search": "search",
    "export": "download",

    # Theme Controls
    "sun": "sun-fill",
    "moon": "moon-stars-fill",

    # UI Controls & Operations
    "file": "file-earmark-text",
    "file-csv": "filetype-csv",
    "settings": "sliders",
    "chart": "bar-chart-line",
    "donut": "pie-chart",
    "check": "check-circle-fill",
    "alert": "exclamation-triangle-fill",
    "info": "info-circle-fill",
    "refresh": "arrow-repeat",
    "process": "cpu",
    "filter": "funnel",
    "browse": "folder2-open",
    "download-cloud": "cloud-download",
    "compress": "arrows-angle-contract",
}


def get_icon(
    name: str,
    size: int = 16,
    className: str = "",
    color: Optional[str] = None,
    aria_label: Optional[str] = None,
) -> html.I:
    """Return a styled, accessible vector icon component.

    :param name: Semantic name from ICON_MAP (e.g. 'dashboard', 'sun', 'moon').
    :param size: Icon font-size in pixels.
    :param className: Additional CSS classes (e.g. 'me-2', 'text-primary').
    :param color: Optional inline color override.
    :param aria_label: Optional accessibility label.
    :return: Dash html.I component.
    """
    icon_class = ICON_MAP.get(name, "app")
    style_dict = {
        "fontSize": f"{size}px",
        "verticalAlign": "middle",
        "display": "inline-flex",
        "alignItems": "center",
    }
    if color:
        style_dict["color"] = color

    return html.I(
        className=f"bi bi-{icon_class} {className}".strip(),
        style=style_dict,
        **({"aria-label": aria_label, "role": "img"} if aria_label else {"aria-hidden": "true"}),
    )


def get_logo_brand(size: int = 32, className: str = "me-2") -> html.Img:
    """Return the original vector LogCluster logo image component."""
    return html.Img(
        src="/assets/icons/logo.svg",
        alt="LogCluster Logo",
        style={
            "width": f"{size}px",
            "height": f"{size}px",
            "verticalAlign": "middle",
        },
        className=className,
    )
