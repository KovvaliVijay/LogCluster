"""
LogCluster — Universal Log Pattern Grouping
Main Web Application (Dash / Flask)

Provides an interactive dashboard for:
  1. Uploading and previewing heterogeneous log files (.log, .txt, .csv)
  2. Unsupervised pattern grouping via Drain template extraction or TF-IDF clustering
  3. Statistical rare-pattern detection with dynamic, real-time threshold adjustment
  4. Drill-down pattern inspection and log exploration
  5. Direct CSV export of enriched results
  6. Complete Light / Dark mode with local storage persistence and vector icon system
"""

import base64
import io
import json
import os
from typing import Optional

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback, dash_table
import pandas as pd
import plotly.express as px

from src.log_loader import LogLoader
from src.log_preprocessor import LogPreprocessor
from src.pattern_grouper import PatternGrouper
from src.rare_pattern_detector import RarePatternDetector
from src.results_exporter import ResultsExporter
from src.icons import get_icon, get_logo_brand

# Initialize Dash application with Bootstrap theme and Bootstrap Icons
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    title="LogCluster — Universal Log Pattern Grouping",
)
server = app.server
app.config["suppress_callback_exceptions"] = True

# Global cache for active session dataset to avoid huge JSON serialization overhead
CURRENT_DATASET = {
    "df": None,           # Annotated DataFrame
    "summary_df": None,   # Pattern summary DataFrame
    "filename": "HealthApp_2000.log",
}

# Pre-populate with verified LogHub dataset on startup
DEFAULT_SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "data", "HealthApp_2000.log")
if os.path.exists(DEFAULT_SAMPLE_PATH):
    try:
        raw_df = LogLoader.load(DEFAULT_SAMPLE_PATH)
        prep = LogPreprocessor()
        clean_df = prep.transform(raw_df)
        grouper = PatternGrouper(method="drain", sim_threshold=0.5)
        grouped_df = grouper.fit_predict(clean_df)
        detector = RarePatternDetector(threshold_value=15.0, threshold_mode="percentage")
        summary = detector.summarize_patterns(grouped_df)
        annotated_df = detector.annotate_records(grouped_df, summary)
        CURRENT_DATASET["df"] = annotated_df
        CURRENT_DATASET["summary_df"] = summary
        CURRENT_DATASET["filename"] = "HealthApp_2000.log"
    except Exception as e:
        print(f"Error loading initial sample data: {e}")


def create_header():
    """Top navigation branding banner with Dark/Light mode toggle."""
    return dbc.Navbar(
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                [
                                    get_logo_brand(36, className="me-2"),
                                    html.Span(
                                        "LogCluster",
                                        className="brand-title",
                                        style={
                                            "fontSize": "1.45rem",
                                        },
                                    ),
                                    html.Span(
                                        " — Universal Log Pattern Grouping",
                                        className="brand-subtitle d-none d-md-inline ms-2",
                                        style={"fontSize": "1.05rem"},
                                    ),
                                ],
                                className="d-flex align-items-center",
                            ),
                            width="auto",
                        ),
                    ],
                    align="center",
                    className="g-0",
                ),
                dbc.Nav(
                    [
                        dbc.NavItem(
                            dbc.Badge(
                                [get_icon("process", 14, "me-1"), "Unsupervised ML"],
                                color="primary",
                                className="p-2 me-2 d-none d-sm-inline-flex align-items-center",
                                style={"fontSize": "0.82rem"},
                            )
                        ),
                        dbc.NavItem(
                            dbc.Badge(
                                [get_icon("rare", 14, "me-1"), "Frequency Rarity"],
                                color="info",
                                className="p-2 me-3 d-none d-sm-inline-flex align-items-center",
                                style={"fontSize": "0.82rem"},
                            )
                        ),
                        dbc.NavItem(
                            dbc.Button(
                                [
                                    html.I(id="theme-toggle-icon", className="bi bi-moon-stars-fill me-2"),
                                    html.Span("Dark Mode", id="theme-toggle-text"),
                                ],
                                id="theme-toggle-btn",
                                className="theme-toggle-btn",
                                title="Toggle between Dark and Light mode",
                                n_clicks=0,
                            )
                        ),
                    ],
                    className="ms-auto align-items-center",
                    navbar=True,
                ),
            ],
            fluid=True,
        ),
        className="navbar-logcluster mb-4 shadow-sm py-2",
    )


# Layout definition
app.layout = html.Div(
    [
        create_header(),
        dbc.Container(
            [
                # Status / Alert notification area
                html.Div(id="alert-container"),
                # Navigation Tabs with Vector Icons
                dbc.Tabs(
                    id="main-tabs",
                    active_tab="tab-dashboard",
                    children=[
                        dbc.Tab(
                            label="Dashboard",
                            label_class_name="tab-icon-dashboard",
                            tab_id="tab-dashboard",
                        ),
                        dbc.Tab(
                            label="Upload & Ingest",
                            label_class_name="tab-icon-upload",
                            tab_id="tab-upload",
                        ),
                        dbc.Tab(
                            label="Pattern Groups",
                            label_class_name="tab-icon-patterns",
                            tab_id="tab-patterns",
                        ),
                        dbc.Tab(
                            label="Rare Patterns",
                            label_class_name="tab-icon-rare",
                            tab_id="tab-rare",
                        ),
                        dbc.Tab(
                            label="Log Explorer",
                            label_class_name="tab-icon-explorer",
                            tab_id="tab-explorer",
                        ),
                        dbc.Tab(
                            label="Export Results",
                            label_class_name="tab-icon-export",
                            tab_id="tab-export",
                        ),
                    ],
                    className="mb-4",
                ),
                # Dynamic tab content area
                html.Div(id="tab-content"),
                # Hidden stores & download
                dcc.Store(id="theme-store", storage_type="local"),
                dcc.Store(id="dataset-version", data=1),
                dcc.Download(id="download-results-csv"),
            ],
            fluid=True,
            style={"maxWidth": "1400px", "paddingBottom": "60px"},
        ),
    ],
    id="app-root",
    style={"minHeight": "100vh"},
)


# ----------------------------------------------------------------------
# Clientside Theme Toggle Callback (Smooth instant toggle + LocalStorage)
# ----------------------------------------------------------------------
app.clientside_callback(
    """
    function(n_clicks, stored_theme) {
        var current = stored_theme;
        if (!current) {
            try {
                current = localStorage.getItem('logcluster-theme');
            } catch(e) {}
        }
        if (!current) {
            if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
                current = 'dark';
            } else {
                current = 'light';
            }
        }

        var next_theme = current;
        var ctx = dash_clientside.callback_context;
        if (ctx && ctx.triggered && ctx.triggered.length > 0 && ctx.triggered[0].prop_id === 'theme-toggle-btn.n_clicks') {
            next_theme = (current === 'dark') ? 'light' : 'dark';
        }

        document.documentElement.setAttribute('data-theme', next_theme);
        if (document.body) {
            document.body.setAttribute('data-theme', next_theme);
        }
        try {
            localStorage.setItem('logcluster-theme', next_theme);
        } catch(e) {}

        var is_dark = (next_theme === 'dark');
        var text = is_dark ? "Light Mode" : "Dark Mode";
        var icon_class = is_dark ? "bi bi-sun-fill me-2 text-warning" : "bi bi-moon-stars-fill me-2 text-primary";

        return [next_theme, text, icon_class];
    }
    """,
    Output("theme-store", "data"),
    Output("theme-toggle-text", "children"),
    Output("theme-toggle-icon", "className"),
    Input("theme-toggle-btn", "n_clicks"),
    State("theme-store", "data"),
    prevent_initial_call=False,
)


# ----------------------------------------------------------------------
# Tab content router
# ----------------------------------------------------------------------
@callback(
    Output("tab-content", "children"),
    Input("main-tabs", "active_tab"),
    Input("dataset-version", "data"),
    Input("theme-store", "data"),
)
def render_tab_content(active_tab, version, theme):
    df = CURRENT_DATASET["df"]
    summary_df = CURRENT_DATASET["summary_df"]
    active_theme = theme or "light"

    if active_tab == "tab-dashboard":
        return build_dashboard_view(df, summary_df, active_theme)
    elif active_tab == "tab-upload":
        return build_upload_view()
    elif active_tab == "tab-patterns":
        return build_patterns_view(summary_df)
    elif active_tab == "tab-rare":
        return build_rare_view(summary_df)
    elif active_tab == "tab-explorer":
        return build_explorer_view(df, summary_df)
    elif active_tab == "tab-export":
        return build_export_view(df, summary_df)
    return html.Div("Select a tab")


# ----------------------------------------------------------------------
# 1. Dashboard View
# ----------------------------------------------------------------------
def build_dashboard_view(df: Optional[pd.DataFrame], summary_df: Optional[pd.DataFrame], theme: str = "light"):
    if df is None or summary_df is None or df.empty:
        return dbc.Alert(
            [get_icon("info", 18, "me-2"), "No log data has been processed yet. Please go to the 'Upload & Ingest' tab to load and process logs."],
            color="warning",
            className="d-flex align-items-center",
        )

    total_logs = len(df)
    total_patterns = len(summary_df)
    rare_count = int(summary_df["is_rare"].sum())
    common_count = total_patterns - rare_count
    current_file = CURRENT_DATASET.get("filename", "Unknown")

    # KPI summary cards with vector icons
    kpi_cards = dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div(
                                [
                                    get_icon("file", 20, "text-primary me-2"),
                                    html.Span("Total Log Entries", className="card-subtitle text-muted"),
                                ],
                                className="d-flex align-items-center mb-2",
                            ),
                            html.H2(f"{total_logs:,}", className="card-title text-primary fw-bold mb-1"),
                            html.Small(f"Loaded from: {current_file}", className="text-muted"),
                        ]
                    ),
                    className="shadow-sm border-0 border-start border-primary border-4",
                ),
                md=3,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div(
                                [
                                    get_icon("patterns", 20, "text-success me-2"),
                                    html.Span("Discovered Patterns", className="card-subtitle text-muted"),
                                ],
                                className="d-flex align-items-center mb-2",
                            ),
                            html.H2(f"{total_patterns:,}", className="card-title text-success fw-bold mb-1"),
                            html.Small("Unsupervised template groups", className="text-muted"),
                        ]
                    ),
                    className="shadow-sm border-0 border-start border-success border-4",
                ),
                md=3,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div(
                                [
                                    get_icon("rare", 20, "text-danger me-2"),
                                    html.Span("Rare Patterns", className="card-subtitle text-muted"),
                                ],
                                className="d-flex align-items-center mb-2",
                            ),
                            html.H2(f"{rare_count:,}", className="card-title text-danger fw-bold mb-1"),
                            html.Small("Infrequent statistical outliers", className="text-muted"),
                        ]
                    ),
                    className="shadow-sm border-0 border-start border-danger border-4",
                ),
                md=3,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div(
                                [
                                    get_icon("compress", 20, "text-info me-2"),
                                    html.Span("Compression Ratio", className="card-subtitle text-muted"),
                                ],
                                className="d-flex align-items-center mb-2",
                            ),
                            html.H2(
                                f"{(1.0 - (total_patterns / max(total_logs, 1))) * 100:.1f}%",
                                className="card-title text-info fw-bold mb-1",
                            ),
                            html.Small(f"{total_logs} logs -> {total_patterns} patterns", className="text-muted"),
                        ]
                    ),
                    className="shadow-sm border-0 border-start border-info border-4",
                ),
                md=3,
            ),
        ],
        className="mb-4",
    )

    # Chart of top patterns with theme support
    is_dark = (theme == "dark")
    chart_template = "plotly_dark" if is_dark else "plotly_white"
    font_color = "#f8fafc" if is_dark else "#0f172a"
    grid_color = "#243049" if is_dark else "#e2e8f0"

    top_n = summary_df.head(10).copy()
    top_n["label"] = top_n["pattern_id"].astype(str) + ": " + top_n["representative_pattern"].str.slice(0, 45) + "..."
    top_n["status"] = top_n["is_rare"].apply(lambda r: "Rare" if r else "Common")

    fig = px.bar(
        top_n.sort_values(by="count", ascending=True),
        x="count",
        y="label",
        orientation="h",
        color="status",
        color_discrete_map={"Common": "#22c55e", "Rare": "#ef4444"},
        title="Top Most Frequent Log Patterns",
        labels={"count": "Occurrence Count", "label": "Pattern Template"},
        height=380,
        template=chart_template,
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Occurrences",
        yaxis_title="",
        legend_title="Classification",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=font_color),
        xaxis=dict(gridcolor=grid_color),
        yaxis=dict(gridcolor=grid_color),
    )

    # Donut chart of common vs rare
    pie_data = pd.DataFrame({
        "Type": ["Common Patterns", "Rare Patterns"],
        "Count": [common_count, rare_count],
    })
    pie_fig = px.pie(
        pie_data,
        names="Type",
        values="Count",
        hole=0.45,
        title="Pattern Distribution",
        color="Type",
        color_discrete_map={"Common Patterns": "#22c55e", "Rare Patterns": "#ef4444"},
        height=380,
        template=chart_template,
    )
    pie_fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=font_color),
    )

    charts_row = dbc.Row(
        [
            dbc.Col(
                dbc.Card(dbc.CardBody(dcc.Graph(figure=fig, config={"displayModeBar": False})), className="shadow-sm border-0"),
                md=8,
            ),
            dbc.Col(
                dbc.Card(dbc.CardBody(dcc.Graph(figure=pie_fig, config={"displayModeBar": False})), className="shadow-sm border-0"),
                md=4,
            ),
        ],
        className="mb-4",
    )

    return html.Div([kpi_cards, charts_row])


# ----------------------------------------------------------------------
# 2. Upload & Ingest View
# ----------------------------------------------------------------------
def build_upload_view():
    return dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div(
                                [
                                    get_icon("file", 18, "text-primary me-2"),
                                    html.Span("1. Select Log Source", className="card-title fw-bold h5 mb-0"),
                                ],
                                className="d-flex align-items-center mb-3",
                            ),
                            # Preset demo dataset selector (Verified LogHub public datasets)
                            html.Label("Verified LogHub Datasets:", className="fw-semibold text-muted mb-1"),
                            dcc.Dropdown(
                                id="preset-selector",
                                options=[
                                    {
                                        "label": "HealthApp System Logs (HealthApp_2000.log) [LogHub]",
                                        "value": "HealthApp_2000.log",
                                    },
                                    {
                                        "label": "BlueGene/L Supercomputer Logs (BGL_2000.log) [LogHub]",
                                        "value": "BGL_2000.log",
                                    },
                                    {
                                        "label": "Hadoop Distributed File System (HDFS_2000.log) [LogHub]",
                                        "value": "HDFS_2000.log",
                                    },
                                ],
                                value="HealthApp_2000.log",
                                clearable=False,
                                className="mb-3",
                            ),
                            html.Div("— OR —", className="text-center text-muted fw-bold mb-3"),
                            # File Upload Drag & Drop
                            dcc.Upload(
                                id="upload-data",
                                children=html.Div(
                                    [
                                        get_icon("upload", 36, "text-primary mb-2"),
                                        html.Div(
                                            ["Drag and Drop or ", html.A("Browse File", className="text-primary fw-bold")],
                                            className="mt-1",
                                        ),
                                        html.Small("Supports .log, .txt, .csv formats", className="text-muted"),
                                    ],
                                    className="p-4 text-center upload-dropzone",
                                ),
                                multiple=False,
                                className="mb-3",
                            ),
                            html.Div(id="upload-filename-display", className="text-muted small mb-3"),
                            # CSV column configuration
                            html.Div(
                                id="csv-column-config",
                                style={"display": "none"},
                                children=[
                                    html.Label("Select Log Message Column (CSV):", className="fw-semibold mb-1"),
                                    dcc.Dropdown(id="csv-column-select", placeholder="Select column containing log text"),
                                    html.Br(),
                                ],
                            ),
                            html.Hr(),
                            html.Div(
                                [
                                    get_icon("settings", 18, "text-primary me-2"),
                                    html.Span("2. Pattern Grouping Settings", className="card-title fw-bold h5 mb-0"),
                                ],
                                className="d-flex align-items-center mb-3",
                            ),
                            html.Label("Algorithm Method:", className="fw-semibold mb-1"),
                            dcc.Dropdown(
                                id="algo-method-select",
                                options=[
                                    {"label": "Drain Algorithm (Recommended: Tree-based template extraction)", "value": "drain"},
                                    {"label": "TF-IDF + KMeans Clustering", "value": "kmeans"},
                                    {"label": "TF-IDF + DBSCAN Density Clustering (with noise handling)", "value": "dbscan"},
                                ],
                                value="drain",
                                clearable=False,
                                className="mb-3",
                            ),
                            html.Div(
                                id="algo-param-container",
                                children=[
                                    html.Label("Similarity Threshold (sim_th):", className="fw-semibold mb-1"),
                                    dcc.Slider(
                                        id="sim-threshold-slider",
                                        min=0.1,
                                        max=0.9,
                                        step=0.05,
                                        value=0.5,
                                        marks={0.1: "0.1", 0.3: "0.3", 0.5: "0.5 (Default)", 0.7: "0.7", 0.9: "0.9"},
                                    ),
                                ],
                            ),
                            html.Hr(),
                            html.Div(
                                [
                                    get_icon("rare", 18, "text-primary me-2"),
                                    html.Span("3. Initial Rarity Threshold", className="card-title fw-bold h5 mb-0"),
                                ],
                                className="d-flex align-items-center mb-3",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Label("Threshold Value:", className="fw-semibold mb-1"),
                                            dbc.Input(id="initial-threshold-val", type="number", value=15.0, min=0.1, max=100.0, step=0.5),
                                        ],
                                        width=6,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Label("Threshold Mode:", className="fw-semibold mb-1"),
                                            dcc.Dropdown(
                                                id="initial-threshold-mode",
                                                options=[
                                                    {"label": "Percentage (<= % of total)", "value": "percentage"},
                                                    {"label": "Absolute Count (<= N logs)", "value": "count"},
                                                ],
                                                value="percentage",
                                                clearable=False,
                                            ),
                                        ],
                                        width=6,
                                    ),
                                ],
                                className="mb-3",
                            ),
                            dbc.Button(
                                [get_icon("process", 18, "me-2"), "Process Logs"],
                                id="process-logs-btn",
                                color="primary",
                                size="lg",
                                className="w-100 fw-bold mt-2 shadow-sm d-flex align-items-center justify-content-center",
                            ),
                        ]
                    ),
                    className="shadow-sm border-0",
                ),
                md=5,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div(
                                [
                                    get_icon("file", 18, "text-secondary me-2"),
                                    html.Span("File Preview & Raw Content", className="card-title fw-bold h5 mb-0"),
                                ],
                                className="d-flex align-items-center mb-3",
                            ),
                            html.Div(
                                id="file-preview-area",
                                children=html.P("Select a file or preset to preview content.", className="text-muted"),
                                className="code-inspector-area",
                                style={
                                    "maxHeight": "520px",
                                    "overflowY": "auto",
                                    "padding": "16px",
                                    "fontSize": "0.88rem",
                                    "whiteSpace": "pre-wrap",
                                },
                            ),
                        ]
                    ),
                    className="shadow-sm border-0",
                ),
                md=7,
            ),
        ]
    )


# File preview callback
@callback(
    Output("file-preview-area", "children"),
    Output("csv-column-config", "style"),
    Output("csv-column-select", "options"),
    Output("upload-filename-display", "children"),
    Input("preset-selector", "value"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
def update_file_preview(preset, upload_contents, upload_filename):
    if upload_contents is not None and upload_filename:
        content_type, content_string = upload_contents.split(",")
        decoded = base64.b64decode(content_string)
        filename_display = f"Uploaded file: {upload_filename} ({len(decoded):,} bytes)"

        if upload_filename.lower().endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(decoded), nrows=10)
                col_options = [{"label": c, "value": c} for c in df.columns]
                preview_text = df.to_string()
                return preview_text, {"display": "block"}, col_options, filename_display
            except Exception as e:
                return f"Error reading uploaded CSV: {e}", {"display": "none"}, [], filename_display
        else:
            text = decoded.decode("utf-8", errors="replace")
            lines = text.splitlines()[:25]
            preview_text = "\n".join(lines)
            return preview_text, {"display": "none"}, [], filename_display

    if preset:
        target_path = os.path.join(os.path.dirname(__file__), "data", preset)
        if os.path.exists(target_path):
            filename_display = f"Using preset: {preset}"
            if preset.endswith(".csv"):
                try:
                    df = pd.read_csv(target_path, nrows=10)
                    col_options = [{"label": c, "value": c} for c in df.columns]
                    preview_text = df.to_string()
                    return preview_text, {"display": "block"}, col_options, filename_display
                except Exception as e:
                    return f"Error loading CSV: {e}", {"display": "none"}, [], filename_display
            else:
                with open(target_path, "r", encoding="utf-8", errors="replace") as f:
                    lines = [f.readline() for _ in range(25)]
                preview_text = "".join(lines)
                return preview_text, {"display": "none"}, [], filename_display

    return "No file selected.", {"display": "none"}, [], ""


# Processing execution callback
@callback(
    Output("dataset-version", "data"),
    Output("alert-container", "children"),
    Input("process-logs-btn", "n_clicks"),
    State("preset-selector", "value"),
    State("upload-data", "contents"),
    State("upload-data", "filename"),
    State("csv-column-select", "value"),
    State("algo-method-select", "value"),
    State("sim-threshold-slider", "value"),
    State("initial-threshold-val", "value"),
    State("initial-threshold-mode", "value"),
    State("dataset-version", "data"),
    prevent_initial_call=True,
)
def process_logs_callback(
    n_clicks,
    preset,
    upload_contents,
    upload_filename,
    csv_col,
    algo_method,
    sim_th,
    init_th_val,
    init_th_mode,
    version,
):
    if not n_clicks:
        return version, dash.no_update

    try:
        source_df = None
        active_filename = "unknown"

        if upload_contents is not None and upload_filename:
            content_type, content_string = upload_contents.split(",")
            decoded = base64.b64decode(content_string)
            buffer = io.BytesIO(decoded)
            source_df = LogLoader.load(buffer, filename=upload_filename, message_column=csv_col)
            active_filename = upload_filename
        elif preset:
            target_path = os.path.join(os.path.dirname(__file__), "data", preset)
            if not os.path.exists(target_path):
                return version, dbc.Alert(f"Selected preset '{preset}' was not found.", color="danger", dismissable=True)

            source_df = LogLoader.load(target_path, message_column=csv_col)
            active_filename = preset
        else:
            return version, dbc.Alert("Please select or upload a log file.", color="warning", dismissable=True)

        if source_df is None or source_df.empty:
            return version, dbc.Alert("No valid log lines found in input file.", color="danger", dismissable=True)

        # 1. Preprocessing & Normalization
        preprocessor = LogPreprocessor()
        clean_df = preprocessor.transform(source_df)

        # 2. Pattern Grouping
        grouper = PatternGrouper(method=algo_method, sim_threshold=sim_th or 0.5)
        grouped_df = grouper.fit_predict(clean_df)

        # 3. Statistical Rarity Identification
        th_val = float(init_th_val) if init_th_val is not None else 15.0
        detector = RarePatternDetector(threshold_value=th_val, threshold_mode=init_th_mode or "percentage")
        summary = detector.summarize_patterns(grouped_df)
        annotated_df = detector.annotate_records(grouped_df, summary)

        # Update cache
        CURRENT_DATASET["df"] = annotated_df
        CURRENT_DATASET["summary_df"] = summary
        CURRENT_DATASET["filename"] = active_filename

        success_msg = dbc.Alert(
            [
                get_icon("check", 18, "me-2"),
                f"Successfully processed {len(annotated_df):,} log lines into {len(summary):,} patterns using {algo_method.upper()}! Identified {summary['is_rare'].sum()} rare patterns.",
            ],
            color="success",
            dismissable=True,
            duration=6000,
            className="d-flex align-items-center",
        )
        return version + 1, success_msg

    except Exception as exc:
        return version, dbc.Alert(f"Error during log processing: {exc}", color="danger", dismissable=True)


# ----------------------------------------------------------------------
# 3. Pattern Groups View
# ----------------------------------------------------------------------
def build_patterns_view(summary_df: Optional[pd.DataFrame]):
    if summary_df is None or summary_df.empty:
        return dbc.Alert(
            [get_icon("info", 18, "me-2"), "No pattern data available. Please process logs first."],
            color="warning",
            className="d-flex align-items-center",
        )

    display_df = summary_df.copy()
    display_df["percentage"] = display_df["percentage"].round(2).astype(str) + "%"
    display_df["status"] = display_df["is_rare"].apply(lambda r: "RARE" if r else "COMMON")

    table_data = display_df.to_dict("records")

    table = dash_table.DataTable(
        id="patterns-datatable",
        columns=[
            {"name": "Pattern ID", "id": "pattern_id"},
            {"name": "Representative Pattern Template", "id": "representative_pattern"},
            {"name": "Occurrences", "id": "count"},
            {"name": "Dataset %", "id": "percentage"},
            {"name": "Status", "id": "status"},
        ],
        data=table_data,
        page_size=12,
        sort_action="native",
        filter_action="native",
        row_selectable="single",
        selected_rows=[0],
        style_cell={"textAlign": "left", "padding": "10px", "fontFamily": "Segoe UI, sans-serif"},
        style_cell_conditional=[
            {"if": {"column_id": "pattern_id"}, "width": "110px", "textAlign": "center"},
            {"if": {"column_id": "count"}, "width": "110px", "textAlign": "center"},
            {"if": {"column_id": "percentage"}, "width": "110px", "textAlign": "center"},
            {"if": {"column_id": "status"}, "width": "120px", "textAlign": "center"},
        ],
        style_data_conditional=[
            {
                "if": {"filter_query": '{status} = "RARE"'},
                "backgroundColor": "var(--bg-table-rare)",
                "color": "var(--text-rare)",
                "fontWeight": "bold",
            }
        ],
    )

    return html.Div(
        [
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        get_icon("patterns", 20, "text-primary me-2"),
                                        html.Span("Discovered Log Patterns Table", className="fw-bold h5 mb-0"),
                                    ],
                                    className="d-flex align-items-center",
                                ),
                                html.P(
                                    "Click any row in the table below to inspect the original raw log messages belonging to that pattern.",
                                    className="text-muted small mt-1 mb-3",
                                ),
                            ]
                        ),
                        table,
                    ]
                ),
                className="shadow-sm border-0 mb-4",
            ),
            # Inspection Drawer / Card
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            [
                                get_icon("search", 18, "text-primary me-2"),
                                html.Span("Logline Inspector (Selected Pattern)", className="fw-bold h5 mb-0"),
                            ],
                            className="d-flex align-items-center mb-2",
                        ),
                        html.Div(id="pattern-loglines-inspection", children="Select a pattern above to view matching loglines."),
                    ]
                ),
                className="shadow-sm border-0",
            ),
        ]
    )


# Pattern row selection drill-down callback
@callback(
    Output("pattern-loglines-inspection", "children"),
    Input("patterns-datatable", "selected_rows"),
    prevent_initial_call=False,
)
def inspect_pattern_loglines(selected_rows):
    summary_df = CURRENT_DATASET["summary_df"]
    df = CURRENT_DATASET["df"]

    if summary_df is None or df is None or not selected_rows:
        return html.P("No pattern selected.", className="text-muted")

    selected_idx = selected_rows[0]
    if selected_idx >= len(summary_df):
        return html.P("Invalid selection.", className="text-muted")

    selected_pid = summary_df.iloc[selected_idx]["pattern_id"]
    rep_template = summary_df.iloc[selected_idx]["representative_pattern"]

    matching_logs = df[df["pattern_id"] == selected_pid]["raw_log"].tolist()

    header_badge = dbc.Badge(
        f"Pattern #{selected_pid}: {len(matching_logs)} matching entries",
        color="primary",
        className="mb-2 p-2",
        style={"fontSize": "0.9rem"},
    )
    template_callout = dbc.Alert(
        [html.B("Template: "), html.Code(rep_template, className="ms-1")],
        color="light",
        className="border mb-3 py-2",
    )

    loglines_display = html.Div(
        [html.Div(f"[{i+1}] {line}", className="mb-1") for i, line in enumerate(matching_logs[:100])],
        className="code-inspector-area",
        style={
            "maxHeight": "320px",
            "overflowY": "auto",
            "padding": "14px",
            "fontSize": "0.85rem",
        },
    )

    return html.Div([header_badge, template_callout, loglines_display])


# ----------------------------------------------------------------------
# 4. Rare Patterns View
# ----------------------------------------------------------------------
def build_rare_view(summary_df: Optional[pd.DataFrame]):
    if summary_df is None or summary_df.empty:
        return dbc.Alert(
            [get_icon("info", 18, "me-2"), "No pattern data available."],
            color="warning",
            className="d-flex align-items-center",
        )

    rare_df = summary_df[summary_df["is_rare"]].copy()
    rare_df["percentage"] = rare_df["percentage"].round(2).astype(str) + "%"

    # Explanation banner
    explanation_card = dbc.Alert(
        [
            html.Div(
                [
                    get_icon("info", 20, "text-info me-2"),
                    html.Span("What is a Rare Pattern?", className="fw-bold h6 mb-0"),
                ],
                className="d-flex align-items-center mb-1",
            ),
            html.P(
                "A log pattern is statistically categorized as 'rare' when its occurrence frequency falls at or below "
                "the configured threshold. Infrequent patterns represent statistical outliers or uncommon operational events. "
                "A rare pattern is NOT automatically an error, security threat, or confirmed system failure.",
                className="mb-0 small",
            ),
        ],
        color="info",
        className="mb-4 shadow-sm",
    )

    # Dynamic Threshold Adjuster Box
    adjuster_card = dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        get_icon("settings", 18, "text-danger me-2"),
                        html.Span("Configure Rarity Frequency Threshold", className="fw-bold h6 mb-0"),
                    ],
                    className="d-flex align-items-center mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Threshold Mode:", className="fw-semibold small text-muted"),
                                dcc.Dropdown(
                                    id="dynamic-rare-mode",
                                    options=[
                                        {"label": "Percentage of Total Dataset (<= X%)", "value": "percentage"},
                                        {"label": "Absolute Occurrence Count (<= N)", "value": "count"},
                                    ],
                                    value="percentage",
                                    clearable=False,
                                ),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                html.Label("Cutoff Threshold Value:", className="fw-semibold small text-muted"),
                                dbc.Input(id="dynamic-rare-threshold", type="number", value=15.0, min=0.1, step=0.5),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                html.Label("Apply Update:", className="fw-semibold small text-muted"),
                                dbc.Button(
                                    [get_icon("refresh", 16, "me-2"), "Update Rarity Threshold"],
                                    id="update-threshold-btn",
                                    color="danger",
                                    className="w-100 fw-bold d-flex align-items-center justify-content-center",
                                ),
                            ],
                            md=4,
                        ),
                    ],
                    align="end",
                ),
                html.Small(
                    "Threshold updates are computed instantly in memory without re-running parsing or clustering.",
                    className="text-muted mt-2 d-block",
                ),
            ]
        ),
        className="shadow-sm border-0 mb-4",
    )

    # Rare table
    rare_table = dash_table.DataTable(
        id="rare-patterns-table",
        columns=[
            {"name": "Pattern ID", "id": "pattern_id"},
            {"name": "Representative Rare Pattern Template", "id": "representative_pattern"},
            {"name": "Occurrences", "id": "count"},
            {"name": "Dataset %", "id": "percentage"},
            {"name": "Sample Raw Logline", "id": "sample_log"},
        ],
        data=rare_df.to_dict("records"),
        page_size=10,
        sort_action="native",
        filter_action="native",
        style_cell={"textAlign": "left", "padding": "10px", "fontFamily": "Segoe UI, sans-serif"},
        style_cell_conditional=[
            {"if": {"column_id": "pattern_id"}, "width": "110px", "textAlign": "center"},
            {"if": {"column_id": "count"}, "width": "110px", "textAlign": "center"},
            {"if": {"column_id": "percentage"}, "width": "110px", "textAlign": "center"},
        ],
    )

    table_card = dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        get_icon("rare", 20, "text-danger me-2"),
                        html.Span(f"Identified Rare Patterns ({len(rare_df)} Total)", className="fw-bold h5 mb-0 text-danger"),
                    ],
                    className="d-flex align-items-center mb-3",
                ),
                rare_table if len(rare_df) > 0 else html.P("No patterns meet the rare threshold criteria.", className="text-muted"),
            ]
        ),
        className="shadow-sm border-0",
    )

    return html.Div([explanation_card, adjuster_card, table_card])


# Dynamic threshold update callback
@callback(
    Output("dataset-version", "data", allow_duplicate=True),
    Output("alert-container", "children", allow_duplicate=True),
    Input("update-threshold-btn", "n_clicks"),
    State("dynamic-rare-threshold", "value"),
    State("dynamic-rare-mode", "value"),
    State("dataset-version", "data"),
    prevent_initial_call=True,
)
def update_rarity_threshold_callback(n_clicks, new_threshold, mode, version):
    if not n_clicks or new_threshold is None:
        return version, dash.no_update

    summary_df = CURRENT_DATASET["summary_df"]
    df = CURRENT_DATASET["df"]

    if summary_df is None or df is None:
        return version, dbc.Alert("No dataset loaded.", color="warning", dismissable=True)

    updated_summary = RarePatternDetector.apply_threshold(summary_df, float(new_threshold), mode)
    annotated_df = RarePatternDetector.annotate_records(df, updated_summary)

    CURRENT_DATASET["summary_df"] = updated_summary
    CURRENT_DATASET["df"] = annotated_df

    rare_count = int(updated_summary["is_rare"].sum())
    return version + 1, dbc.Alert(
        [
            get_icon("check", 18, "me-2"),
            f"Rarity threshold updated to {new_threshold} ({mode}). Found {rare_count} rare patterns instantly!",
        ],
        color="info",
        dismissable=True,
        duration=4000,
        className="d-flex align-items-center",
    )


# ----------------------------------------------------------------------
# 5. Log Explorer View
# ----------------------------------------------------------------------
def build_explorer_view(df: Optional[pd.DataFrame], summary_df: Optional[pd.DataFrame]):
    if df is None or df.empty:
        return dbc.Alert(
            [get_icon("info", 18, "me-2"), "No logs to explore."],
            color="warning",
            className="d-flex align-items-center",
        )

    pattern_options = [{"label": "All Patterns", "value": "ALL"}]
    if summary_df is not None:
        for _, row in summary_df.iterrows():
            pid = row["pattern_id"]
            pattern_options.append({"label": f"Pattern {pid} ({row['count']} logs)", "value": str(pid)})

    filter_controls = dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        get_icon("filter", 18, "text-primary me-2"),
                        html.Span("Log Explorer & Search Filters", className="fw-bold h6 mb-0"),
                    ],
                    className="d-flex align-items-center mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Filter by Pattern ID:", className="fw-semibold small text-muted"),
                                dcc.Dropdown(id="explorer-pattern-filter", options=pattern_options, value="ALL", clearable=False),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                html.Label("Filter by Rarity Status:", className="fw-semibold small text-muted"),
                                dcc.Dropdown(
                                    id="explorer-rarity-filter",
                                    options=[
                                        {"label": "All Records", "value": "ALL"},
                                        {"label": "Rare Only", "value": "RARE"},
                                        {"label": "Common Only", "value": "COMMON"},
                                    ],
                                    value="ALL",
                                    clearable=False,
                                ),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                html.Label("Search Text in Message:", className="fw-semibold small text-muted"),
                                dbc.Input(id="explorer-text-search", placeholder="Type keyword to filter...", type="text"),
                            ],
                            md=4,
                        ),
                    ]
                )
            ]
        ),
        className="shadow-sm border-0 mb-4",
    )

    table_container = html.Div(id="explorer-table-container")

    return html.Div([filter_controls, table_container])


# Log explorer filtering callback
@callback(
    Output("explorer-table-container", "children"),
    Input("explorer-pattern-filter", "value"),
    Input("explorer-rarity-filter", "value"),
    Input("explorer-text-search", "value"),
    Input("dataset-version", "data"),
)
def update_explorer_table(pattern_filter, rarity_filter, search_text, version):
    df = CURRENT_DATASET["df"]
    if df is None or df.empty:
        return html.P("No data available.", className="text-muted")

    filtered = df.copy()

    if pattern_filter and pattern_filter != "ALL":
        filtered = filtered[filtered["pattern_id"].astype(str) == str(pattern_filter)]

    if rarity_filter == "RARE":
        filtered = filtered[filtered["is_rare"] == True]
    elif rarity_filter == "COMMON":
        filtered = filtered[filtered["is_rare"] == False]

    if search_text:
        filtered = filtered[filtered["raw_log"].str.contains(search_text, case=False, na=False)]

    filtered_count = len(filtered)
    export_df = filtered[["line_number", "pattern_id", "is_rare", "representative_pattern", "raw_log"]].copy()
    export_df["is_rare"] = export_df["is_rare"].apply(lambda r: "RARE" if r else "COMMON")

    table = dash_table.DataTable(
        columns=[
            {"name": "Line #", "id": "line_number"},
            {"name": "Pattern ID", "id": "pattern_id"},
            {"name": "Status", "id": "is_rare"},
            {"name": "Representative Template", "id": "representative_pattern"},
            {"name": "Original Raw Message", "id": "raw_log"},
        ],
        data=export_df.head(500).to_dict("records"),
        page_size=15,
        sort_action="native",
        style_cell={"textAlign": "left", "padding": "8px", "fontFamily": "Segoe UI, sans-serif"},
        style_cell_conditional=[
            {"if": {"column_id": "line_number"}, "width": "80px", "textAlign": "center"},
            {"if": {"column_id": "pattern_id"}, "width": "100px", "textAlign": "center"},
            {"if": {"column_id": "is_rare"}, "width": "100px", "textAlign": "center"},
            {"if": {"column_id": "representative_pattern"}, "width": "300px"},
        ],
        style_data_conditional=[
            {
                "if": {"filter_query": '{is_rare} = "RARE"'},
                "backgroundColor": "var(--bg-table-rare)",
                "color": "var(--text-rare)",
                "fontWeight": "bold",
            }
        ],
    )

    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.Span(f"Showing {min(filtered_count, 500)} of {filtered_count} matching records", className="fw-semibold text-muted"),
                    ],
                    className="mb-2",
                ),
                table,
            ]
        ),
        className="shadow-sm border-0",
    )


# ----------------------------------------------------------------------
# 6. Export Results View
# ----------------------------------------------------------------------
def build_export_view(df: Optional[pd.DataFrame], summary_df: Optional[pd.DataFrame]):
    if df is None or df.empty:
        return dbc.Alert(
            [get_icon("info", 18, "me-2"), "No processed data available for export."],
            color="warning",
            className="d-flex align-items-center",
        )

    total_logs = len(df)
    total_patterns = len(summary_df) if summary_df is not None else 0
    rare_count = int(summary_df["is_rare"].sum()) if summary_df is not None else 0

    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        get_icon("export", 22, "text-success me-2"),
                        html.Span("Export Grouped Log Results", className="fw-bold h5 mb-0"),
                    ],
                    className="d-flex align-items-center mb-3",
                ),
                html.P(
                    "Download the complete analysis dataset containing verbatim raw log messages, "
                    "discovered pattern IDs, representative templates, occurrence counts, dataset percentages, "
                    "and rarity status.",
                    className="text-muted",
                ),
                dbc.ListGroup(
                    [
                        dbc.ListGroupItem([get_icon("file", 16, "me-2 text-primary"), f"Total Log Records: {total_logs:,}"]),
                        dbc.ListGroupItem([get_icon("patterns", 16, "me-2 text-success"), f"Discovered Patterns: {total_patterns:,}"]),
                        dbc.ListGroupItem([get_icon("rare", 16, "me-2 text-danger"), f"Identified Rare Patterns: {rare_count:,}"]),
                        dbc.ListGroupItem([get_icon("file-csv", 16, "me-2 text-info"), "Export Format: Standard CSV (RFC 4180 compliant)"]),
                    ],
                    className="mb-4",
                    style={"maxWidth": "450px"},
                ),
                dbc.Button(
                    [get_icon("download-cloud", 20, "me-2"), "Download Results CSV"],
                    id="btn-download-csv",
                    color="success",
                    size="lg",
                    className="fw-bold shadow-sm d-flex align-items-center",
                ),
            ]
        ),
        className="shadow-sm border-0",
    )


# Download CSV action callback
@callback(
    Output("download-results-csv", "data"),
    Input("btn-download-csv", "n_clicks"),
    prevent_initial_call=True,
)
def handle_csv_download(n_clicks):
    if not n_clicks:
        return dash.no_update

    df = CURRENT_DATASET["df"]
    if df is None or df.empty:
        return dash.no_update

    export_df = ResultsExporter.prepare_export_df(df)
    return dcc.send_data_frame(export_df.to_csv, "logcluster_results.csv", index=False)


# Main server entrypoint
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=False)
