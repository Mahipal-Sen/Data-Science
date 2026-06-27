"""
=============================================================================
FLIGHT FARE PREDICTION ENGINE - STREAMLIT WEB APPLICATION
=============================================================================
Role: Main entry point for the flight fare prediction system
Purpose: Streamlit web app with separate pages for:
  - Data loading and model training
  - Dataset insights and exploratory analysis
  - Real-time fare prediction

Usage: streamlit run app.py
=============================================================================
"""

import base64
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import preprocess
import train
import utils


st.set_page_config(
    page_title='Flight Fare Predictor',
    page_icon=':airplane:',
    layout='wide',
    initial_sidebar_state='expanded'
)

utils.setup_logging()


PAGE_DATA = 'Data Loading'
PAGE_INSIGHTS = 'Insights'
PAGE_PREDICTION = 'Prediction'

APP_NAME = 'Flight Fare Intelligence'
APP_TAGLINE = 'A real-world pricing console for route demand, booking windows, and ML-powered fare estimates.'
PAGE_DETAILS = {
    PAGE_DATA: 'Upload, validate, and train against the active flight fare dataset.',
    PAGE_INSIGHTS: 'Explore dataset quality, fare patterns, category trends, and route behavior.',
    PAGE_PREDICTION: 'Estimate a fare from flight details with time and demand adjustments.'
}

PROJECT_ROOT = Path(__file__).resolve().parent
HERO_IMAGE_PATH = PROJECT_ROOT / 'assets' / 'flight_fare_command_center.png'

CHART_BG = '#071421'
CHART_PANEL = '#0d1b2a'
CHART_GRID = '#21384e'
CHART_TEXT = '#f8fafc'
CHART_MUTED = '#a7b8c9'
CHART_PRIMARY = '#14b8a6'
CHART_ACCENT = '#f59e0b'
CHART_BLUE = '#38bdf8'
CHART_ROSE = '#fb7185'


@st.cache_data(show_spinner=False)
def get_asset_data_uri(path: str) -> str:
    asset_path = Path(path)
    if not asset_path.exists():
        return ''

    encoded = base64.b64encode(asset_path.read_bytes()).decode('ascii')
    return f'data:image/png;base64,{encoded}'


def inject_app_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --app-bg: #06111e;
            --panel: #0d1b2a;
            --panel-soft: #102338;
            --panel-muted: #132a43;
            --ink: #071421;
            --border: #20344a;
            --border-strong: #2d4a64;
            --text: #f8fafc;
            --muted: #a7b6c8;
            --subtle: #7f93a8;
            --primary: #14b8a6;
            --primary-dark: #0f9488;
            --blue: #38bdf8;
            --amber: #f59e0b;
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #fb7185;
            --violet: #8b5cf6;
            --shadow: 0 22px 54px rgba(0, 0, 0, 0.34);
            --shadow-soft: 0 12px 28px rgba(0, 0, 0, 0.22);
        }

        .stApp {
            background:
                radial-gradient(circle at top right, rgba(20, 184, 166, 0.14), transparent 30rem),
                linear-gradient(180deg, #071421 0%, var(--app-bg) 46%, #050c15 100%);
            color: var(--text);
        }

        .block-container {
            max-width: 1280px;
            padding-top: 1.2rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            color: var(--text);
            letter-spacing: 0;
        }

        h2 {
            font-size: 1.32rem;
            font-weight: 800;
            margin-top: 1.4rem;
        }

        h3 {
            font-size: 1rem;
            font-weight: 760;
            margin-top: 1.15rem;
        }

        p, label, .stMarkdown, .stCaption, .stText, span {
            color: var(--text);
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, #071421 0%, #0a1b2c 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        [data-testid="stSidebar"] > div {
            padding-top: 1.3rem;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #f8fafc;
        }

        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label {
            color: #e6edf7 !important;
        }

        .brand-panel {
            border-bottom: 1px solid rgba(255, 255, 255, 0.12);
            padding: 0.15rem 0 1.1rem 0;
            margin-bottom: 1rem;
        }

        .brand-title {
            font-size: 1.12rem;
            font-weight: 850;
            color: #ffffff;
        }

        .brand-copy {
            color: #b8c6d8;
            font-size: 0.82rem;
            line-height: 1.35;
            margin-top: 0.25rem;
        }

        .sidebar-status {
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.06);
            padding: 0.8rem;
            margin-top: 0.8rem;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
        }

        .sidebar-label {
            color: #7dd3fc;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0;
            text-transform: uppercase;
        }

        .sidebar-value {
            color: #f8fafc;
            font-size: 0.9rem;
            font-weight: 700;
            margin-top: 0.15rem;
            overflow-wrap: anywhere;
        }

        .app-header {
            position: relative;
            display: flex;
            align-items: flex-end;
            min-height: 255px;
            background-color: #0b1c2e;
            background-position: center right;
            background-size: cover;
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 8px;
            box-shadow: var(--shadow);
            padding: 1.75rem 1.8rem;
            margin-bottom: 1rem;
            overflow: hidden;
        }

        .app-header::before {
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(90deg, rgba(7, 20, 33, 0.96) 0%, rgba(7, 20, 33, 0.84) 42%, rgba(7, 20, 33, 0.10) 100%);
            pointer-events: none;
        }

        .app-header::after {
            content: "";
            position: absolute;
            inset: 0 0 auto 0;
            height: 4px;
            background: linear-gradient(90deg, #14b8a6, #38bdf8, #f97316);
            pointer-events: none;
        }

        .app-hero-copy {
            position: relative;
            z-index: 1;
            max-width: 720px;
        }

        .app-eyebrow {
            color: #7dd3fc;
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .app-title {
            color: #ffffff;
            font-size: clamp(2.05rem, 4vw, 3.15rem);
            font-weight: 900;
            line-height: 1.08;
            max-width: 680px;
        }

        .app-subtitle {
            color: #dce8f5;
            font-size: 1rem;
            line-height: 1.5;
            max-width: 660px;
            margin-top: 0.55rem;
        }

        .hero-micro-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(130px, 1fr));
            gap: 0.65rem;
            max-width: 650px;
            margin-top: 1rem;
        }

        .hero-chip {
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.10);
            padding: 0.72rem 0.8rem;
        }

        .hero-chip span {
            display: block;
            color: #a7c5df !important;
            font-size: 0.72rem;
            font-weight: 740;
            text-transform: uppercase;
        }

        .hero-chip strong {
            display: block;
            color: #ffffff;
            font-size: 1rem;
            margin-top: 0.16rem;
        }

        .page-heading {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin: 1.1rem 0 0.65rem 0;
        }

        .page-heading-title {
            color: var(--text);
            font-size: 1.32rem;
            font-weight: 800;
        }

        .page-heading-copy {
            color: var(--muted);
            font-size: 0.88rem;
            text-align: right;
        }

        .status-strip {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 0.75rem;
            margin: 0 0 1.05rem 0;
        }

        .status-tile {
            background: linear-gradient(180deg, #0f2236 0%, #0a1928 100%);
            border: 1px solid var(--border);
            border-left: 4px solid var(--blue);
            border-radius: 8px;
            padding: 0.82rem 0.95rem;
            box-shadow: var(--shadow-soft);
        }

        .status-tile.good {
            border-left-color: var(--success);
        }

        .status-tile.warn {
            border-left-color: var(--warning);
        }

        .status-tile.neutral {
            border-left-color: var(--border-strong);
        }

        .status-title {
            color: var(--subtle);
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0;
            text-transform: uppercase;
        }

        .status-number {
            color: var(--text);
            font-size: 1rem;
            font-weight: 800;
            margin-top: 0.18rem;
            overflow-wrap: anywhere;
        }

        .work-note {
            border: 1px solid var(--border);
            border-left: 4px solid var(--primary);
            border-radius: 8px;
            background: linear-gradient(180deg, #0f2236 0%, #0b1c2e 100%);
            color: var(--muted);
            padding: 0.9rem 1rem;
            line-height: 1.45;
            box-shadow: var(--shadow-soft);
        }

        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, #0f2236 0%, #0a1928 100%);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.9rem 1rem;
            box-shadow: var(--shadow-soft);
        }

        div[data-testid="stMetric"] label {
            color: var(--muted) !important;
            font-weight: 720;
        }

        div[data-testid="stMetricValue"] {
            color: var(--text);
            font-weight: 820;
        }

        div[data-testid="stMetricValue"] * {
            color: var(--text) !important;
        }

        .stButton button {
            border-radius: 8px;
            font-weight: 750;
            border: 1px solid var(--border-strong);
            background: #0f2236;
            color: var(--text);
            transition: all 120ms ease;
            min-height: 2.55rem;
        }

        .stButton button:hover {
            border-color: var(--primary);
            color: var(--primary);
        }

        .stButton button[kind="primary"] {
            background: linear-gradient(135deg, var(--primary) 0%, #0ea5e9 100%);
            border-color: var(--primary);
            color: #ffffff;
            box-shadow: 0 10px 24px rgba(15, 118, 110, 0.22);
        }

        .stButton button[kind="primary"]:hover {
            background: linear-gradient(135deg, var(--primary-dark) 0%, #0284c7 100%);
            border-color: var(--primary-dark);
            color: #ffffff;
        }

        div[data-testid="stFileUploader"] {
            background: #0b1c2e;
            border: 1px dashed var(--border-strong);
            border-radius: 8px;
            padding: 0.85rem;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #0d1b2a;
            border-color: var(--border) !important;
            box-shadow: var(--shadow-soft);
        }

        div[data-testid="stDataFrame"],
        div[data-testid="stTable"] {
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
            border-bottom: 1px solid var(--border);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 0.45rem 0.75rem;
            font-weight: 720;
            color: var(--muted);
        }

        .stTabs [aria-selected="true"] {
            color: var(--primary) !important;
            border-bottom-color: var(--primary) !important;
        }

        .stAlert {
            border-radius: 8px;
            border: 1px solid var(--border);
            background: #0f2236;
        }

        input, textarea, [data-baseweb="select"] > div {
            background-color: #0b1c2e !important;
            border-color: var(--border-strong) !important;
            color: var(--text) !important;
            border-radius: 8px !important;
        }

        input:focus, textarea:focus {
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 1px var(--primary) !important;
        }

        [data-baseweb="popover"] {
            background: #0d1b2a !important;
            border: 1px solid var(--border) !important;
        }

        [data-baseweb="menu"] {
            background: #0d1b2a !important;
        }

        [role="option"] {
            color: var(--text) !important;
        }

        [data-testid="stExpander"] {
            background: #0d1b2a;
            border: 1px solid var(--border);
            border-radius: 8px;
        }

        .route-preview {
            border: 1px solid var(--border);
            border-radius: 8px;
            background:
                radial-gradient(circle at top right, rgba(56, 189, 248, 0.16), transparent 18rem),
                linear-gradient(135deg, #0f2236 0%, #0b1c2e 62%, #0c2b36 100%);
            padding: 1rem;
            margin-top: 0.9rem;
            box-shadow: var(--shadow-soft);
        }

        .route-topline {
            display: flex;
            justify-content: space-between;
            gap: 0.8rem;
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
        }

        .route-topline strong {
            color: var(--primary);
            overflow-wrap: anywhere;
        }

        .route-path {
            display: grid;
            grid-template-columns: minmax(120px, 1fr) minmax(170px, 1.2fr) minmax(120px, 1fr);
            align-items: center;
            gap: 0.85rem;
            margin: 1rem 0 0.8rem 0;
        }

        .airport span {
            display: block;
            color: var(--subtle) !important;
            font-size: 0.72rem;
            font-weight: 800;
            text-transform: uppercase;
        }

        .airport strong {
            display: block;
            color: var(--text);
            font-size: clamp(1.25rem, 2.1vw, 1.9rem);
            line-height: 1.05;
            overflow-wrap: anywhere;
        }

        .airport small {
            color: var(--muted);
            font-weight: 740;
        }

        .airport.align-right {
            text-align: right;
        }

        .route-line {
            display: grid;
            grid-template-columns: 10px 1fr 34px 1fr 10px;
            align-items: center;
            gap: 0.35rem;
        }

        .route-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--primary);
            box-shadow: 0 0 0 5px rgba(15, 118, 110, 0.12);
        }

        .route-dash {
            height: 2px;
            background: repeating-linear-gradient(90deg, var(--border-strong), var(--border-strong) 10px, transparent 10px, transparent 16px);
        }

        .route-plane {
            display: grid;
            place-items: center;
            width: 34px;
            height: 34px;
            border-radius: 50%;
            background: #14b8a6;
            color: #ffffff !important;
            font-size: 1rem;
            font-weight: 800;
            box-shadow: 0 8px 18px rgba(15, 118, 110, 0.22);
        }

        .route-meta {
            display: grid;
            grid-template-columns: repeat(3, minmax(120px, 1fr));
            gap: 0.55rem;
        }

        .route-meta-item {
            border-top: 1px solid var(--border);
            padding-top: 0.58rem;
        }

        .route-meta-item span {
            display: block;
            color: var(--subtle) !important;
            font-size: 0.72rem;
            font-weight: 800;
            text-transform: uppercase;
        }

        .route-meta-item strong {
            display: block;
            color: var(--text);
            font-size: 0.94rem;
            margin-top: 0.08rem;
            overflow-wrap: anywhere;
        }

        .fare-result-panel {
            border-radius: 8px;
            border: 1px solid rgba(20, 184, 166, 0.30);
            background:
                radial-gradient(circle at top right, rgba(245, 158, 11, 0.13), transparent 18rem),
                linear-gradient(135deg, #071421 0%, #0b2135 58%, #0d3440 100%);
            color: #ffffff;
            padding: 1.25rem;
            margin-top: 1.1rem;
            box-shadow: 0 22px 48px rgba(7, 20, 33, 0.22);
            overflow: hidden;
        }

        .fare-result-panel * {
            color: inherit;
        }

        .fare-result-head {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 1rem;
        }

        .fare-label {
            color: #7dd3fc !important;
            font-size: 0.76rem;
            font-weight: 850;
            text-transform: uppercase;
        }

        .fare-price {
            color: #ffffff;
            font-size: clamp(2rem, 4.5vw, 3.7rem);
            font-weight: 920;
            line-height: 1;
            margin-top: 0.28rem;
        }

        .fare-subtitle {
            color: #d7e7f4 !important;
            font-size: 0.94rem;
            margin-top: 0.38rem;
            max-width: 640px;
        }

        .fare-badge {
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.10);
            color: #ffffff !important;
            font-size: 0.78rem;
            font-weight: 800;
            padding: 0.42rem 0.68rem;
            white-space: nowrap;
        }

        .fare-result-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(130px, 1fr));
            gap: 0.72rem;
            margin-top: 1rem;
        }

        .fare-tile {
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.08);
            padding: 0.85rem;
        }

        .fare-tile span {
            display: block;
            color: #a9c4da !important;
            font-size: 0.72rem;
            font-weight: 800;
            text-transform: uppercase;
        }

        .fare-tile strong {
            display: block;
            color: #ffffff;
            font-size: 1.05rem;
            margin-top: 0.18rem;
        }

        .fare-footnote {
            border-top: 1px solid rgba(255, 255, 255, 0.13);
            color: #c8d9e8 !important;
            font-size: 0.84rem;
            line-height: 1.45;
            margin-top: 1rem;
            padding-top: 0.8rem;
        }

        [data-testid="stToolbar"] {
            opacity: 0.35;
        }

        @media (max-width: 900px) {
            .page-heading {
                align-items: flex-start;
                flex-direction: column;
            }

            .page-heading-copy {
                text-align: left;
            }

            .hero-micro-grid,
            .route-meta,
            .fare-result-grid {
                grid-template-columns: 1fr;
            }

            .route-path {
                grid-template-columns: 1fr;
            }

            .route-line {
                min-height: 34px;
            }

            .airport.align-right {
                text-align: left;
            }

            .fare-result-head {
                flex-direction: column;
            }
        }

        @media (max-width: 560px) {
            .app-header {
                padding: 1rem;
                min-height: 310px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def configure_chart_theme() -> None:
    plt.rcParams.update({
        'axes.facecolor': CHART_PANEL,
        'figure.facecolor': CHART_BG,
        'savefig.facecolor': CHART_BG,
        'axes.edgecolor': CHART_GRID,
        'axes.labelcolor': CHART_TEXT,
        'axes.titlecolor': CHART_TEXT,
        'axes.titleweight': 'bold',
        'grid.color': CHART_GRID,
        'xtick.color': CHART_MUTED,
        'ytick.color': CHART_MUTED,
        'text.color': CHART_TEXT,
        'font.size': 10
    })


def style_chart_axes(ax) -> None:
    ax.set_facecolor(CHART_PANEL)
    ax.grid(axis='x', color=CHART_GRID, alpha=0.55, linewidth=0.8)
    ax.grid(axis='y', color=CHART_GRID, alpha=0.2, linewidth=0.6)
    ax.tick_params(colors=CHART_MUTED)
    ax.xaxis.label.set_color(CHART_MUTED)
    ax.yaxis.label.set_color(CHART_MUTED)
    ax.title.set_color(CHART_TEXT)

    for spine in ax.spines.values():
        spine.set_color(CHART_GRID)


def render_bar_chart(series: pd.Series, title: str, color: str = CHART_PRIMARY) -> None:
    if series is None or series.empty:
        st.info('No chart data available.')
        return

    chart_data = series.dropna().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(3.0, min(6.5, 0.42 * len(chart_data) + 1.4))))
    ax.barh(chart_data.index.astype(str), chart_data.values, color=color, alpha=0.9)
    ax.set_title(title)
    style_chart_axes(ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def load_saved_model(show_warning: bool = True):
    try:
        model = utils.load_pipeline()
        metadata = utils.load_metadata()
        return model, metadata
    except Exception as exc:
        if show_warning:
            st.warning('No trained model found. Train a model from the Data Loading page.')
            st.write(str(exc))
        return None, {}


def get_current_model(show_warning: bool = True):
    model = st.session_state.get('model')
    metadata = st.session_state.get('metadata')

    if model is not None and metadata:
        return model, metadata

    model, metadata = load_saved_model(show_warning=show_warning)
    if model is not None:
        st.session_state.model = model
        st.session_state.metadata = metadata

    return model, metadata


def read_uploaded_dataset(uploaded_file) -> pd.DataFrame:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'csv':
        raw_df = pd.read_csv(uploaded_file)
    else:
        raw_df = pd.read_excel(uploaded_file)

    index_columns = {'index', 'unnamed: 0', 'unnamed:_0'}
    columns_to_drop = [
        col for col in raw_df.columns
        if str(col).strip().lower() in index_columns
    ]

    if columns_to_drop:
        raw_df = raw_df.drop(columns=columns_to_drop)

    return raw_df


def prepare_dataset(raw_df: pd.DataFrame, show_warning: bool = True) -> pd.DataFrame:
    try:
        return utils.clean_price_column(raw_df)
    except Exception as exc:
        if show_warning:
            st.warning(f'Price cleaning warning: {exc}')
        return raw_df


def refresh_dataset_state() -> None:
    raw_df = st.session_state.get('raw_df')

    if raw_df is None:
        st.session_state.dataset_profile = {}
        st.session_state.dataset_ready = False
        st.session_state.dataset_message = 'No dataset loaded.'
        return

    profile = preprocess.infer_dataset_profile(raw_df)
    dataset_ready, dataset_message = preprocess.validate_ready_for_training(raw_df)

    st.session_state.dataset_profile = profile
    st.session_state.dataset_ready = dataset_ready
    st.session_state.dataset_message = dataset_message


def set_current_dataset(raw_df: pd.DataFrame, source_name: str, show_warning: bool = True) -> None:
    st.session_state.raw_df = prepare_dataset(raw_df, show_warning=show_warning)
    st.session_state.dataset_source = source_name
    refresh_dataset_state()


def initialize_session_state() -> None:
    if 'raw_df' not in st.session_state:
        set_current_dataset(
            utils.get_demo_dataset(),
            'Default demo dataset',
            show_warning=False
        )


def get_loaded_dataset() -> Optional[pd.DataFrame]:
    raw_df = st.session_state.get('raw_df')
    if raw_df is None or raw_df.empty:
        return None
    return raw_df


def get_dataset_profile() -> dict:
    profile = st.session_state.get('dataset_profile')
    if not profile and get_loaded_dataset() is not None:
        refresh_dataset_state()
        profile = st.session_state.get('dataset_profile', {})
    return profile or {}


def render_status_strip() -> None:
    raw_df = get_loaded_dataset()
    dataset_ready = st.session_state.get('dataset_ready', False)

    if raw_df is None:
        dataset_value = 'No dataset'
        shape_value = 'Waiting for upload'
        source_value = 'Not loaded'
        dataset_tone = 'neutral'
        readiness_value = 'Not ready'
        readiness_tone = 'neutral'
    else:
        dataset_value = st.session_state.get('dataset_source', 'Active dataset')
        shape_value = f'{raw_df.shape[0]:,} rows x {raw_df.shape[1]:,} columns'
        source_value = 'Loaded'
        dataset_tone = 'good'
        readiness_value = 'Ready for training' if dataset_ready else 'Needs review'
        readiness_tone = 'good' if dataset_ready else 'warn'

    _, metadata = get_current_model(show_warning=False)
    model_name = metadata.get('best_model') if metadata else None
    model_value = model_name or 'Not loaded'
    model_tone = 'good' if model_name else 'neutral'

    st.markdown(
        f"""
        <div class="status-strip">
            <div class="status-tile {dataset_tone}">
                <div class="status-title">Dataset</div>
                <div class="status-number">{escape(str(source_value))}</div>
            </div>
            <div class="status-tile">
                <div class="status-title">Source</div>
                <div class="status-number">{escape(str(dataset_value))}</div>
            </div>
            <div class="status-tile {readiness_tone}">
                <div class="status-title">Readiness</div>
                <div class="status-number">{escape(str(readiness_value))}</div>
            </div>
            <div class="status-tile {model_tone}">
                <div class="status-title">Model</div>
                <div class="status-number">{escape(str(model_value))}</div>
            </div>
            <div class="status-tile neutral">
                <div class="status-title">Dataset Shape</div>
                <div class="status-number">{escape(str(shape_value))}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_work_note(text: str) -> None:
    st.markdown(
        f'<div class="work-note">{escape(text)}</div>',
        unsafe_allow_html=True
    )


def render_header(page_title: str) -> None:
    page_detail = PAGE_DETAILS.get(page_title, '')
    hero_uri = get_asset_data_uri(str(HERO_IMAGE_PATH))
    hero_style = f' style="background-image: url({hero_uri});"' if hero_uri else ''
    st.markdown(
        f"""
        <div class="app-header"{hero_style}>
            <div class="app-hero-copy">
                <div class="app-eyebrow">Airline revenue intelligence</div>
                <div class="app-title">{escape(APP_NAME)}</div>
                <div class="app-subtitle">{escape(APP_TAGLINE)}</div>
                <div class="hero-micro-grid">
                    <div class="hero-chip">
                        <span>Pricing mode</span>
                        <strong>Route aware</strong>
                    </div>
                    <div class="hero-chip">
                        <span>Market signal</span>
                        <strong>Demand tuned</strong>
                    </div>
                    <div class="hero-chip">
                        <span>Output</span>
                        <strong>Live fare view</strong>
                    </div>
                </div>
            </div>
        </div>
        <div class="page-heading">
            <div class="page-heading-title">{escape(page_title)}</div>
            <div class="page-heading-copy">{escape(page_detail)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    render_status_strip()


def render_sidebar() -> str:
    st.sidebar.markdown(
        f"""
        <div class="brand-panel">
            <div class="brand-title">{escape(APP_NAME)}</div>
            <div class="brand-copy">Dataset management, fare insights, and prediction in one workspace.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    page = st.sidebar.radio(
        'Workspace',
        [PAGE_DATA, PAGE_INSIGHTS, PAGE_PREDICTION],
    )

    raw_df = get_loaded_dataset()
    dataset_source = st.session_state.get('dataset_source', 'Unknown source')
    if raw_df is None:
        dataset_value = 'No dataset loaded'
        shape_value = 'No active table'
    else:
        dataset_value = dataset_source
        shape_value = f'{raw_df.shape[0]:,} rows x {raw_df.shape[1]:,} columns'

    model, metadata = get_current_model(show_warning=False)
    if model is None:
        model_value = 'Not loaded'
    else:
        model_value = metadata.get('best_model', 'Saved model')

    st.sidebar.markdown(
        f"""
        <div class="sidebar-status">
            <div class="sidebar-label">Active dataset</div>
            <div class="sidebar-value">{escape(str(dataset_value))}</div>
        </div>
        <div class="sidebar-status">
            <div class="sidebar-label">Table size</div>
            <div class="sidebar-value">{escape(str(shape_value))}</div>
        </div>
        <div class="sidebar-status">
            <div class="sidebar-label">Model</div>
            <div class="sidebar-value">{escape(str(model_value))}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    return page


def show_dataset_preview(df: pd.DataFrame) -> None:
    st.subheader('Dataset preview')
    st.dataframe(df.head(10), use_container_width=True)
    with st.expander('Dataset statistics'):
        st.write(df.describe(include='all').transpose())


def show_price_metrics(df: pd.DataFrame) -> None:
    try:
        cleaned_df = utils.clean_price_column(df)
        stats = utils.get_price_statistics(cleaned_df)
        st.subheader('Price metrics')
        col1, col2, col3 = st.columns(3)
        col1.metric('Mean price', f"Rs. {stats['mean']:,.2f}" if stats['mean'] is not None else 'N/A')
        col2.metric('Median price', f"Rs. {stats['median']:,.2f}" if stats['median'] is not None else 'N/A')
        col3.metric('Mode price', f"Rs. {stats['mode']:,.2f}" if stats['mode'] is not None else 'N/A')
    except Exception as exc:
        st.warning(f'Price statistics unavailable: {exc}')


def show_profile(profile: dict) -> None:
    st.subheader('Detected dataset profile')
    st.markdown(f"**Detected target:** {profile.get('target')}")
    st.markdown(f"**Required features missing:** {', '.join(profile.get('missing_required', [])) or 'None'}")
    st.markdown('**Available columns:**')
    st.write(profile.get('available_columns'))


def render_dataset_status(df: pd.DataFrame) -> None:
    dataset_ready = st.session_state.get('dataset_ready', False)
    dataset_message = st.session_state.get('dataset_message', 'Dataset status unavailable.')

    col1, col2, col3 = st.columns(3)
    col1.metric('Rows', f'{df.shape[0]:,}')
    col2.metric('Columns', f'{df.shape[1]:,}')
    col3.metric('Training status', 'Ready' if dataset_ready else 'Needs fixes')

    if dataset_ready:
        st.success(dataset_message)
    else:
        st.warning(dataset_message)
        st.info('Required columns: Airline, Source, Destination, Date_of_Journey, Dep_Time, Duration, and a Price/Fare field.')


def render_model_training(df: pd.DataFrame) -> None:
    st.subheader('Model training')
    dataset_ready = st.session_state.get('dataset_ready', False)

    if st.button('Train model on current dataset', type='primary', disabled=not dataset_ready):
        with st.spinner('Training models... This may take 30-60 seconds.'):
            try:
                report = train.train_model(df, save_model=True)
                trained_metadata = dict(report['metadata'])
                trained_metadata.update({
                    'best_model': report['best_model'],
                    'metrics': report['metrics'],
                    'candidates': report['candidates']
                })

                st.session_state.model = report['pipeline']
                st.session_state.metadata = trained_metadata

                st.success('Model training complete.')
                st.write('Best model:', report['best_model'])
                plot_model_performance(report['metrics'])
                st.write('Candidate models and metrics:')
                st.json(report['candidates'])
            except Exception as exc:
                st.error(f'Failed to train model: {exc}')

    if not dataset_ready:
        st.caption('Training is disabled until the dataset contains all required fields.')


def render_data_loading_page() -> None:
    render_header(PAGE_DATA)

    render_work_note('Start by loading a CSV or Excel dataset. The app will inspect the schema, clean the fare column when possible, and make the same dataset available to Insights and Prediction.')

    col1, col2 = st.columns([2, 1])
    with col1:
        with st.container(border=True):
            st.subheader('Dataset source')
            uploaded_file = st.file_uploader('Upload CSV or Excel dataset', type=['csv', 'xlsx', 'xls'])
            if uploaded_file is not None:
                try:
                    uploaded_df = read_uploaded_dataset(uploaded_file)
                    set_current_dataset(uploaded_df, uploaded_file.name)
                    st.success(f'Loaded dataset from {uploaded_file.name}.')
                except Exception as exc:
                    st.error(f'Unable to load dataset: {exc}')

            if st.button('Load default demo dataset'):
                set_current_dataset(utils.get_demo_dataset(), 'Default demo dataset', show_warning=False)
                st.success('Demo dataset loaded.')

    with col2:
        with st.container(border=True):
            st.subheader('Workflow')
            st.markdown('1. Load a dataset')
            st.markdown('2. Review readiness')
            st.markdown('3. Train or retrain the model')
            st.markdown('4. Move to prediction')

    raw_df = get_loaded_dataset()
    if raw_df is None:
        st.warning('Upload a dataset or load the demo dataset to continue.')
        return

    render_dataset_status(raw_df)
    st.subheader('Current dataset')
    st.dataframe(raw_df.head(10), use_container_width=True)

    with st.expander('Detected profile'):
        show_profile(get_dataset_profile())

    render_model_training(raw_df)


def render_insights_page() -> None:
    render_header(PAGE_INSIGHTS)

    raw_df = get_loaded_dataset()
    if raw_df is None:
        st.warning('Load a dataset from the Data Loading page first.')
        return

    insight_df = preprocess.standardize_columns(raw_df)
    profile = preprocess.infer_dataset_profile(insight_df)

    render_dataset_overview(insight_df)
    render_column_quality(insight_df)
    render_price_insights(insight_df, profile)
    render_categorical_insights(insight_df, profile)
    render_route_insights(insight_df, profile)
    show_dataset_preview(insight_df)


def render_predict_now_card() -> None:
    st.markdown('### Fare quote console')
    render_work_note(
        'Route, schedule, airline, and stopover context feed the ML estimate, booking-window adjustment, '
        'and simulated live market fare.'
    )


def format_rupees(value: float) -> str:
    return f'Rs. {value:,.0f}'


def render_route_preview(
    source: str,
    destination: str,
    airline: str,
    dep_date,
    dep_time: str,
    arrival_time: str,
    duration: str,
    stops: str
) -> None:
    journey_date = dep_date.strftime('%d %b %Y') if hasattr(dep_date, 'strftime') else str(dep_date)
    st.markdown(
        f"""
        <div class="route-preview">
            <div class="route-topline">
                <span>Fare search preview</span>
                <strong>{escape(str(airline))}</strong>
            </div>
            <div class="route-path">
                <div class="airport">
                    <span>From</span>
                    <strong>{escape(str(source))}</strong>
                    <small>{escape(str(dep_time))}</small>
                </div>
                <div class="route-line">
                    <span class="route-dot"></span>
                    <span class="route-dash"></span>
                    <span class="route-plane">&#9992;</span>
                    <span class="route-dash"></span>
                    <span class="route-dot"></span>
                </div>
                <div class="airport align-right">
                    <span>To</span>
                    <strong>{escape(str(destination))}</strong>
                    <small>{escape(str(arrival_time))}</small>
                </div>
            </div>
            <div class="route-meta">
                <div class="route-meta-item">
                    <span>Journey</span>
                    <strong>{escape(journey_date)}</strong>
                </div>
                <div class="route-meta-item">
                    <span>Duration</span>
                    <strong>{escape(str(duration))}</strong>
                </div>
                <div class="route-meta-item">
                    <span>Stops</span>
                    <strong>{escape(str(stops))}</strong>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def _input_value(raw_input: dict, roles: Dict[str, Optional[str]], key: str, fallback: str) -> str:
    return str(raw_input.get(_role_name(roles, key, fallback), ''))


def render_prediction_result(
    raw_input: dict,
    roles: Dict[str, Optional[str]],
    base_prediction: float,
    adjusted_price: float,
    live_price: float,
    days_left: int,
    trend_factor: float,
    demand_factor: float
) -> None:
    source = _input_value(raw_input, roles, 'source', 'Source')
    destination = _input_value(raw_input, roles, 'destination', 'Destination')
    airline = _input_value(raw_input, roles, 'airline', 'Airline')
    dep_time = _input_value(raw_input, roles, 'dep_time', 'Dep_Time')
    route_label = f'{source} to {destination}' if source or destination else 'Selected route'
    live_delta = ((live_price - base_prediction) / base_prediction * 100) if base_prediction else 0

    if days_left < 3:
        booking_signal = 'Last-minute pressure'
    elif days_left > 20:
        booking_signal = 'Advance booking'
    else:
        booking_signal = 'Standard window'

    st.markdown(
        f"""
        <div class="fare-result-panel">
            <div class="fare-result-head">
                <div>
                    <div class="fare-label">Recommended live fare</div>
                    <div class="fare-price">{escape(format_rupees(live_price))}</div>
                    <div class="fare-subtitle">
                        {escape(airline)} fare estimate for {escape(route_label)} with departure around {escape(dep_time)}.
                    </div>
                </div>
                <div class="fare-badge">{escape(booking_signal)}</div>
            </div>
            <div class="fare-result-grid">
                <div class="fare-tile">
                    <span>Base model</span>
                    <strong>{escape(format_rupees(base_prediction))}</strong>
                </div>
                <div class="fare-tile">
                    <span>Time adjusted</span>
                    <strong>{escape(format_rupees(adjusted_price))}</strong>
                </div>
                <div class="fare-tile">
                    <span>Live change</span>
                    <strong>{live_delta:+.1f}%</strong>
                </div>
                <div class="fare-tile">
                    <span>Days left</span>
                    <strong>{days_left}</strong>
                </div>
            </div>
            <div class="fare-footnote">
                Trend factor {trend_factor:.2f} and demand factor {demand_factor:.2f} were applied to convert the ML baseline into a market-style fare.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def _role_name(roles: Dict[str, Optional[str]], key: str, fallback: str) -> str:
    return roles.get(key) or fallback


def _options_for_role(df: pd.DataFrame, roles: Dict[str, Optional[str]], key: str, fallback_options: list[str]) -> list[str]:
    role_col = roles.get(key)
    if role_col in df.columns:
        values = df[role_col].dropna().astype(str).unique().tolist()
        if values:
            return sorted(values)
    return fallback_options


def create_input_fields(
    df: pd.DataFrame,
    option_roles: Dict[str, Optional[str]],
    input_roles: Optional[Dict[str, Optional[str]]] = None
) -> dict:
    st.subheader('Flight details')

    input_roles = input_roles or option_roles

    airline_options = _options_for_role(df, option_roles, 'airline', ['IndiGo', 'Air India', 'SpiceJet', 'Jet Airways'])
    source_options = _options_for_role(df, option_roles, 'source', ['Delhi', 'Banglore', 'Mumbai', 'Kolkata'])
    destination_options = _options_for_role(df, option_roles, 'destination', ['New Delhi', 'Cochin', 'Hyderabad', 'Chennai'])
    stops_options = _options_for_role(df, option_roles, 'stops', ['non-stop', '1 stop', '2 stops'])

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            airline = st.selectbox('Airline', airline_options)
            source = st.selectbox('Source City', source_options)
            dep_date = st.date_input('Journey Date', value=datetime.today())
            dep_time = st.text_input('Departure Time (HH:MM)', value='09:00')

        with col2:
            destination = st.selectbox('Destination City', destination_options)
            duration = st.text_input('Duration', value='2h 30m')
            arrival_time = st.text_input('Arrival Time (HH:MM)', value='11:30')
            stops = st.selectbox('Total Stops', stops_options)

        with col3:
            route = st.text_input('Route', value=f'{source} -> {destination}')
            additional_info = st.text_input('Additional Info', value='No info')
            st.caption('Times use HH:MM format. Duration can be entered as 2h 30m.')

        route = route.strip() or f'{source} -> {destination}'
        render_route_preview(
            source=source,
            destination=destination,
            airline=airline,
            dep_date=dep_date,
            dep_time=dep_time,
            arrival_time=arrival_time,
            duration=duration,
            stops=stops
        )

    return {
        _role_name(input_roles, 'airline', 'Airline'): airline,
        _role_name(input_roles, 'source', 'Source'): source,
        _role_name(input_roles, 'destination', 'Destination'): destination,
        _role_name(input_roles, 'date', 'Date_of_Journey'): dep_date.strftime('%d/%m/%Y'),
        _role_name(input_roles, 'dep_time', 'Dep_Time'): dep_time,
        _role_name(input_roles, 'arrival_time', 'Arrival_Time'): arrival_time,
        _role_name(input_roles, 'duration', 'Duration'): duration,
        _role_name(input_roles, 'stops', 'Total_Stops'): stops,
        _role_name(input_roles, 'route', 'Route'): route,
        _role_name(input_roles, 'additional_info', 'Additional_Info'): additional_info
    }


def plot_model_performance(metrics: dict) -> None:
    st.subheader('Model evaluation')
    col1, col2, col3 = st.columns(3)
    col1.metric('R2 score', f"{metrics.get('r2', 0):.3f}")
    col2.metric('MAE', f"{metrics.get('mae', 0):.2f}")
    col3.metric('RMSE', f"{metrics.get('rmse', 0):.2f}")


def _price_series(df: pd.DataFrame, profile: dict) -> tuple[Optional[str], pd.Series]:
    price_col = profile.get('target')
    if price_col not in df.columns:
        return None, pd.Series(dtype=float)

    series = pd.to_numeric(df[price_col], errors='coerce').dropna()
    return price_col, series


def render_dataset_overview(df: pd.DataFrame) -> None:
    st.subheader('Dataset overview')

    total_cells = int(df.shape[0] * df.shape[1])
    missing_cells = int(df.isna().sum().sum())
    missing_percent = (missing_cells / total_cells * 100) if total_cells else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Rows', f'{df.shape[0]:,}')
    col2.metric('Columns', f'{df.shape[1]:,}')
    col3.metric('Duplicate rows', f'{df.duplicated().sum():,}')
    col4.metric('Missing cells', f'{missing_percent:.1f}%')

    numeric_count = len(df.select_dtypes(include='number').columns)
    categorical_count = len(df.select_dtypes(exclude='number').columns)
    col5, col6 = st.columns(2)
    col5.metric('Numeric columns', f'{numeric_count:,}')
    col6.metric('Categorical/text columns', f'{categorical_count:,}')


def render_column_quality(df: pd.DataFrame) -> None:
    st.subheader('Column quality')

    quality_df = pd.DataFrame({
        'column': df.columns,
        'type': [str(dtype) for dtype in df.dtypes],
        'non_null': df.notna().sum().values,
        'missing': df.isna().sum().values,
        'missing_percent': (df.isna().mean().values * 100).round(2),
        'unique_values': df.nunique(dropna=True).values
    })

    st.dataframe(quality_df, use_container_width=True)

    missing_df = quality_df[quality_df['missing'] > 0].sort_values('missing', ascending=False)
    if missing_df.empty:
        st.success('No missing values found in the loaded dataset.')
    else:
        st.markdown('**Columns with missing values**')
        render_bar_chart(
            missing_df.set_index('column')['missing'],
            'Missing values by column',
            color=CHART_ROSE
        )


def render_price_insights(df: pd.DataFrame, profile: dict) -> None:
    st.subheader('Fare insights')

    price_col, series = _price_series(df, profile)
    if price_col is None or series.empty:
        st.info('No numeric fare/price column was detected for fare insights.')
        return

    mode_series = series.mode()
    mode_value = mode_series.iloc[0] if not mode_series.empty else None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Minimum fare', f'Rs. {series.min():,.0f}')
    col2.metric('Average fare', f'Rs. {series.mean():,.0f}')
    col3.metric('Median fare', f'Rs. {series.median():,.0f}')
    col4.metric('Maximum fare', f'Rs. {series.max():,.0f}')

    if mode_value is not None:
        st.caption(f'Most common fare: Rs. {mode_value:,.0f}')

    col5, col6 = st.columns(2)
    with col5:
        fig, ax = plt.subplots()
        series.plot.hist(
            bins=25,
            alpha=0.86,
            ax=ax,
            color=CHART_PRIMARY,
            edgecolor=CHART_BG,
            linewidth=0.8
        )
        ax.set_title('Fare distribution')
        ax.set_xlabel(price_col)
        ax.set_ylabel('Flights')
        style_chart_axes(ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col6:
        fig, ax = plt.subplots()
        ax.boxplot(
            series,
            vert=True,
            patch_artist=True,
            boxprops={'facecolor': CHART_BLUE, 'color': CHART_GRID, 'alpha': 0.78},
            medianprops={'color': CHART_ACCENT, 'linewidth': 2},
            whiskerprops={'color': CHART_MUTED},
            capprops={'color': CHART_MUTED},
            flierprops={
                'marker': 'o',
                'markerfacecolor': CHART_ROSE,
                'markeredgecolor': CHART_ROSE,
                'alpha': 0.72
            }
        )
        ax.set_title('Fare spread')
        ax.set_ylabel(price_col)
        ax.set_xticklabels(['Fares'])
        style_chart_axes(ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def render_categorical_insights(df: pd.DataFrame, profile: dict) -> None:
    st.subheader('Category insights')

    roles = profile.get('roles', {})
    price_col, series = _price_series(df, profile)
    available_roles = [
        ('Airlines', roles.get('airline')),
        ('Source cities', roles.get('source')),
        ('Destination cities', roles.get('destination')),
        ('Stops', roles.get('stops'))
    ]
    visible_roles = [(label, col) for label, col in available_roles if col in df.columns]

    if not visible_roles:
        st.info('No airline, source, destination, or stops columns were detected.')
        return

    count_tabs = st.tabs([label for label, _ in visible_roles])
    for tab_index, (label, col) in enumerate(visible_roles):
        with count_tabs[tab_index]:
            counts = df[col].astype(str).value_counts().head(10)
            st.markdown(f'**Top {label.lower()} by records**')
            render_bar_chart(counts, f'Top {label.lower()} by records', color=CHART_BLUE)

            if price_col is not None and not series.empty:
                price_by_category = (
                    df.assign(_price=pd.to_numeric(df[price_col], errors='coerce'))
                    .dropna(subset=['_price'])
                    .groupby(col)['_price']
                    .mean()
                    .sort_values(ascending=False)
                    .head(10)
                )
                st.markdown(f'**Average fare by {label.lower()}**')
                render_bar_chart(price_by_category, f'Average fare by {label.lower()}', color=CHART_ACCENT)


def render_route_insights(df: pd.DataFrame, profile: dict) -> None:
    st.subheader('Route insights')

    roles = profile.get('roles', {})
    source_col = roles.get('source')
    destination_col = roles.get('destination')
    price_col, series = _price_series(df, profile)

    if source_col not in df.columns or destination_col not in df.columns:
        st.info('Source and destination columns are needed for route insights.')
        return

    route_df = df.copy()
    route_df['_route'] = route_df[source_col].astype(str) + ' -> ' + route_df[destination_col].astype(str)

    col1, col2 = st.columns(2)
    with col1:
        route_counts = route_df['_route'].value_counts().head(10)
        st.markdown('**Most common routes**')
        render_bar_chart(route_counts, 'Most common routes', color=CHART_BLUE)

    with col2:
        if price_col is None or series.empty:
            st.info('Average fare by route needs a numeric fare column.')
            return

        route_df['_price'] = pd.to_numeric(route_df[price_col], errors='coerce')
        route_prices = (
            route_df.dropna(subset=['_price'])
            .groupby('_route')['_price']
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )
        st.markdown('**Highest average fare routes**')
        render_bar_chart(route_prices, 'Highest average fare routes', color=CHART_ACCENT)


def render_prediction_page() -> None:
    render_header(PAGE_PREDICTION)

    raw_df = get_loaded_dataset()
    if raw_df is None:
        st.warning('Load a dataset from the Data Loading page first.')
        return

    model, metadata = get_current_model(show_warning=True)
    if model is None:
        st.info('Train a model on the Data Loading page before making predictions.')
        return

    profile = get_dataset_profile()
    dataset_ready = st.session_state.get('dataset_ready', False)
    if not dataset_ready:
        st.warning('Prediction is disabled because the loaded dataset is missing required feature columns.')
        st.info('Use the Data Loading page to load a compatible dataset.')
        return

    option_df = preprocess.standardize_columns(raw_df)
    model_roles = metadata.get('roles') or profile.get('roles') or {}
    option_roles = preprocess.infer_roles(option_df) or profile.get('roles') or model_roles

    render_predict_now_card()
    raw_input = create_input_fields(option_df, option_roles, model_roles)

    if st.button('Predict fare', type='primary'):
        with st.spinner('Running prediction...'):
            try:
                X_input = preprocess.preprocess_prediction_data(raw_input, model_roles)
                base_prediction = float(model.predict(X_input)[0])
                days_left = int(X_input.at[0, 'days_left']) if 'days_left' in X_input.columns else 0
                trend_factor = float(X_input.at[0, 'trend_factor']) if 'trend_factor' in X_input.columns else 1.0
                adjusted_price = utils.adjust_price_for_time(base_prediction, days_left, trend_factor)
                demand_factor = float(np.random.uniform(0.9, 1.3))
                live_price = utils.get_live_price(
                    base_prediction,
                    days_left,
                    trend_factor,
                    demand_factor=demand_factor
                )

                render_prediction_result(
                    raw_input=raw_input,
                    roles=model_roles,
                    base_prediction=base_prediction,
                    adjusted_price=adjusted_price,
                    live_price=live_price,
                    days_left=days_left,
                    trend_factor=trend_factor,
                    demand_factor=demand_factor
                )

                with st.expander('Price adjustment details'):
                    st.write('Base ML prediction:', format_rupees(base_prediction))
                    st.write('Days left before journey:', days_left)
                    st.write('Yearly trend factor:', f'{trend_factor:.2f}')
                    st.write('Demand factor:', f'{demand_factor:.2f}')
                    st.write('Adjusted price after time factors:', format_rupees(adjusted_price))
                    st.write('Final simulated live price:', format_rupees(live_price))
            except Exception as exc:
                st.error(f'Prediction failed: {exc}')


def main() -> None:
    inject_app_styles()
    configure_chart_theme()
    initialize_session_state()
    page = render_sidebar()

    if page == PAGE_DATA:
        render_data_loading_page()
    elif page == PAGE_INSIGHTS:
        render_insights_page()
    elif page == PAGE_PREDICTION:
        render_prediction_page()


if __name__ == '__main__':
    main()
