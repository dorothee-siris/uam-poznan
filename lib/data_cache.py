# lib/data_cache.py — Cached loaders for UAM parquet files
#
# FILENAME MAPPING (actual files differ from SCHEMAS.md):
#   thematic_overview.parquet       ← same
#   thematic_sublevels.parquet      ← SCHEMAS says "thematic_detail_sublevels"
#   thematic_partners.parquet       ← SCHEMAS says "thematic_detail_partners"
#   thematic_authors.parquet        ← SCHEMAS says "thematic_detail_authors"
#   thematic_treemap.parquet        ← SCHEMAS says "treemap_hierarchy"
#   all_topics.parquet              ← same

import os
import streamlit as st
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _load(filename: str) -> pd.DataFrame | None:
    """Load a parquet file from the data directory, returning None if missing."""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


# ── Thematic data ────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_thematic_overview() -> pd.DataFrame | None:
    return _load("thematic_overview.parquet")


@st.cache_data(show_spinner=False)
def load_thematic_detail_sublevels() -> pd.DataFrame | None:
    return _load("thematic_sublevels.parquet")


@st.cache_data(show_spinner=False)
def load_thematic_detail_partners() -> pd.DataFrame | None:
    return _load("thematic_partners.parquet")


@st.cache_data(show_spinner=False)
def load_thematic_detail_authors() -> pd.DataFrame | None:
    return _load("thematic_authors.parquet")


@st.cache_data(show_spinner=False)
def load_treemap_hierarchy() -> pd.DataFrame | None:
    return _load("thematic_treemap.parquet")


# ── Taxonomy (shared, constant) ──────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_all_topics() -> pd.DataFrame | None:
    return _load("all_topics.parquet")


# ── Convenience accessors ────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_overview_for_level(level: str) -> pd.DataFrame:
    """Return rows from thematic_overview matching the given level."""
    df = load_thematic_overview()
    if df is None:
        return pd.DataFrame()
    return df[df["level"] == level].copy()


@st.cache_data(show_spinner=False)
def get_sublevels_for_parent(parent_level: str, parent_id: str) -> pd.DataFrame:
    """Return child rows from thematic_sublevels for a given parent."""
    df = load_thematic_detail_sublevels()
    if df is None:
        return pd.DataFrame()
    mask = (df["parent_level"] == parent_level) & (df["parent_id"] == str(parent_id))
    return df[mask].copy()


@st.cache_data(show_spinner=False)
def get_partners_for_element(level: str, element_id: str) -> pd.Series | None:
    """Return the partner row for a given thematic element."""
    df = load_thematic_detail_partners()
    if df is None:
        return None
    mask = (df["level"] == level) & (df["id"] == str(element_id))
    rows = df[mask]
    if rows.empty:
        return None
    return rows.iloc[0]


@st.cache_data(show_spinner=False)
def get_authors_for_element(level: str, element_id: str) -> pd.Series | None:
    """Return the author row for a given thematic element."""
    df = load_thematic_detail_authors()
    if df is None:
        return None
    mask = (df["level"] == level) & (df["id"] == str(element_id))
    rows = df[mask]
    if rows.empty:
        return None
    return rows.iloc[0]
