# lib/helpers.py — Formatters, parsers, taxonomy helpers, and chart factories

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from lib.config import (
    DOMAIN_COLORS,
    DOMAIN_EMOJI,
    DOMAIN_NAMES_ORDERED,
    DOMAIN_ORDER,
    SHORT_CODE,
)


# =====================================================================
#  Safe value converters
# =====================================================================

def safe_int(v: Any, default: int = 0) -> int:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return default
    try:
        return int(str(v).replace(",", "").strip())
    except (ValueError, TypeError):
        return default


def safe_float(v: Any, default: float = 0.0) -> float:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return default
    try:
        return float(str(v).replace(",", "").strip())
    except (ValueError, TypeError):
        return default


# =====================================================================
#  Formatters
# =====================================================================

def format_int(v: Any) -> str:
    n = safe_int(v)
    return f"{n:,}"


def format_pct(v: Any, decimals: int = 1) -> str:
    f = safe_float(v)
    if f == 0.0:
        return "—"
    return f"{f * 100:.{decimals}f}%"


def format_cagr(v: Any) -> str:
    f = safe_float(v)
    if f == 0.0:
        return "—"
    sign = "+" if f > 0 else ""
    return f"{sign}{f * 100:.1f}%"


def format_fwci(v: Any) -> str:
    f = safe_float(v)
    if f == 0.0:
        return "—"
    return f"{f:.2f}"


def format_si(v: Any) -> str:
    f = safe_float(v)
    if f == 0.0:
        return "—"
    return f"{f:.2f}"


# =====================================================================
#  Blob parsers
# =====================================================================

def _is_empty(blob: Any) -> bool:
    """Check if a blob value is missing or empty."""
    if blob is None:
        return True
    if isinstance(blob, float) and math.isnan(blob):
        return True
    if isinstance(blob, str) and blob.strip() == "":
        return True
    return False


def parse_kv_blob(blob: str) -> dict[str, str]:
    """Parse 'key:value|key:value|...' → dict."""
    if _is_empty(blob):
        return {}
    result = {}
    for pair in str(blob).split("|"):
        pair = pair.strip()
        if ":" not in pair:
            continue
        k, v = pair.split(":", 1)
        result[k.strip()] = v.strip()
    return result


def parse_year_counts(blob: str) -> dict[int, int]:
    """Parse 'YYYY:N|YYYY:N|...' → {year: count}."""
    kv = parse_kv_blob(blob)
    return {safe_int(k): safe_int(v) for k, v in kv.items() if k}


def parse_domain_counts(blob: str) -> dict[int, int]:
    """Parse '1:N|2:N|3:N|4:N' → {domain_id: count}."""
    kv = parse_kv_blob(blob)
    return {safe_int(k): safe_int(v) for k, v in kv.items() if k}


def parse_pipe_float_list(blob: str) -> list[float]:
    """Parse 'v1|v2|v3|...' → list of floats."""
    if _is_empty(blob):
        return []
    return [safe_float(x) for x in str(blob).split("|")]


def parse_fwci_boxplot(blob: str) -> dict[str, float] | None:
    """Parse 7-value FWCI boxplot blob → dict with p0..p100."""
    vals = parse_pipe_float_list(blob)
    if len(vals) != 7:
        return None
    keys = ["p0", "p10", "p25", "p50", "p75", "p90", "p100"]
    return dict(zip(keys, vals))


def parse_top_items(blob: str, fields: list[str]) -> list[dict]:
    """
    Parse colon-record blobs:
      'id:name:country:type:copubs:share_inst:share_int:share_partner:fwci|...'
    Returns list of dicts keyed by `fields`.
    """
    if _is_empty(blob):
        return []
    items = []
    for record in str(blob).split("|"):
        record = record.strip()
        if not record:
            continue
        parts = record.split(":")
        item = {}
        for i, field_name in enumerate(fields):
            item[field_name] = parts[i].strip() if i < len(parts) else ""
        items.append(item)
    return items


def parse_sdg_breakdown(blob: str) -> dict[int, int]:
    """Parse SDG breakdown blob → {sdg_id: count}."""
    return {safe_int(k): safe_int(v) for k, v in parse_kv_blob(blob).items() if k}


# =====================================================================
#  Partner blob parsing (specific field lists)
# =====================================================================

INT_PARTNER_FIELDS = [
    "id", "name", "country", "type", "copubs",
    "share_inst", "share_int", "share_partner", "fwci",
]

DOMESTIC_PARTNER_FIELDS = [
    "id", "name", "type", "copubs",
    "share_inst", "share_int", "share_partner", "fwci",
]

RECIPROCITY_PARTNER_FIELDS = [
    "id", "name", "country", "type", "copubs",
    "share_inst", "share_int", "share_partner", "partner_total", "fwci",
]

AUTHOR_FIELDS = [
    "id", "name", "orcid", "pubs", "pct", "fwci", "is_inst", "labs",
]


def parse_int_partners(blob: str) -> list[dict]:
    return parse_top_items(blob, INT_PARTNER_FIELDS)


def parse_domestic_partners(blob: str) -> list[dict]:
    return parse_top_items(blob, DOMESTIC_PARTNER_FIELDS)


def parse_reciprocity_partners(blob: str) -> list[dict]:
    return parse_top_items(blob, RECIPROCITY_PARTNER_FIELDS)


def parse_authors(blob: str) -> list[dict]:
    return parse_top_items(blob, AUTHOR_FIELDS)


# =====================================================================
#  Taxonomy helpers
# =====================================================================

def domain_name_for_id(domain_id: int | str) -> str:
    mapping = {1: "Life Sciences", 2: "Social Sciences",
               3: "Physical Sciences", 4: "Health Sciences"}
    return mapping.get(safe_int(domain_id), "Other")


def domain_color(name_or_id) -> str:
    return DOMAIN_COLORS.get(name_or_id, DOMAIN_COLORS.get("Other", "#7f7f7f"))


def domain_emoji(name: str) -> str:
    return DOMAIN_EMOJI.get(name, "⬜")


def get_col(row, preferred: str, *fallbacks, default=None):
    """Try to get a column value from a row / dict, with fallbacks."""
    for col in (preferred, *fallbacks):
        try:
            val = row[col] if isinstance(row, dict) else row.get(col)
        except (KeyError, TypeError):
            val = None
        if val is not None and not (isinstance(val, float) and math.isnan(val)):
            return val
    return default


def pubs_pct_col() -> str:
    """Return the institution-specific pubs percentage column name."""
    return f"pubs_pct_of_{SHORT_CODE}"


# =====================================================================
#  Domain color legend (HTML)
# =====================================================================

def render_domain_legend() -> str:
    """Return inline HTML for the domain color legend."""
    items = []
    for name in DOMAIN_NAMES_ORDERED:
        color = DOMAIN_COLORS[name]
        items.append(
            f'<span style="display:inline-flex;align-items:center;margin-right:14px;">'
            f'<span style="width:12px;height:12px;border-radius:2px;'
            f'background:{color};display:inline-block;margin-right:4px;"></span>'
            f'<span style="font-size:0.85rem;">{name}</span></span>'
        )
    return "".join(items)


# =====================================================================
#  Chart factory functions  (see VIZ_CATALOG.md)
# =====================================================================

def plot_horizontal_bar(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
    text_col: str | None = None,
    color_col: str | None = None,
    fixed_color: str = "#4e79a7",
    title: str = "",
    xaxis_title: str = "Publications",
) -> go.Figure:
    """VIZ-01: Horizontal bar with optional % labels."""
    df = df.sort_values(x_col, ascending=True)  # largest at top after y-reversal
    colors = df[color_col].tolist() if color_col and color_col in df.columns else fixed_color
    text = df[text_col].tolist() if text_col and text_col in df.columns else None

    fig = go.Figure(go.Bar(
        y=df[y_col].tolist(),
        x=df[x_col].tolist(),
        orientation="h",
        marker_color=colors,
        text=text,
        textposition="auto",
    ))
    fig.update_layout(
        template="plotly_white",
        height=max(300, len(df) * 35 + 80),
        margin=dict(t=30, l=10, r=10, b=10),
        xaxis_title=xaxis_title,
        yaxis_title="",
        title=title,
    )
    return fig


def plot_precomputed_boxplot(
    boxplot_data: list[dict],
    show_extremes: bool = False,
    orientation: str = "v",
    height: int = 500,
) -> go.Figure:
    """VIZ-02: Boxplot from pre-aggregated percentiles."""
    fig = go.Figure()
    for item in boxplot_data:
        lower = item["p0"] if show_extremes else item["p10"]
        upper = item["p100"] if show_extremes else item["p90"]
        cat = item.get("category", "")
        color = item.get("color", "#4e79a7")
        if orientation == "v":
            fig.add_trace(go.Box(
                x=[cat], lowerfence=[lower], q1=[item["p25"]],
                median=[item["p50"]], q3=[item["p75"]], upperfence=[upper],
                marker_color=color, fillcolor=color,
                line=dict(color=color, width=1.5),
                boxpoints=False, showlegend=False,
            ))
            fig.add_annotation(
                x=cat, y=-0.03, yref="paper",
                text=f'{item.get("count", 0):,}',
                showarrow=False, font=dict(size=10, color="#666"),
            )
        else:
            fig.add_trace(go.Box(
                y=[cat], lowerfence=[lower], q1=[item["p25"]],
                median=[item["p50"]], q3=[item["p75"]], upperfence=[upper],
                orientation="h",
                marker_color=color, fillcolor=color,
                line=dict(color=color, width=1.5),
                boxpoints=False, showlegend=False,
            ))

    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(t=30, l=50, r=30, b=50),
    )
    return fig


def plot_treemap(
    df: pd.DataFrame,
    color_metric: str = "fwci_median",
    range_color: list | None = None,
    color_continuous_scale: str | None = None,
    height: int = 600,
) -> go.Figure:
    """VIZ-04: Hierarchical treemap."""
    scale_map = {
        "fwci_median": ("RdYlGn", [0, 2]),
        "pct_international": ("Blues", None),
        "cagr": ("RdYlGn", [-0.2, 0.2]),
    }
    if color_continuous_scale is None:
        color_continuous_scale = scale_map.get(color_metric, ("Blues", None))[0]
    if range_color is None:
        range_color = scale_map.get(color_metric, (None, None))[1]

    fig = px.treemap(
        df,
        ids="id",
        names="name",
        parents="parent_id",
        values="pubs",
        color=color_metric,
        color_continuous_scale=color_continuous_scale,
        range_color=range_color,
    )
    fig.update_traces(branchvalues="total", maxdepth=3, tiling=dict(pad=1))
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(t=30, l=10, r=10, b=10),
    )
    return fig


def plot_time_series(
    df: pd.DataFrame,
    year_col: str = "Year",
    name_col: str = "Name",
    count_col: str = "Count",
    top_n: int = 10,
    height: int = 400,
) -> tuple[go.Figure, go.Figure]:
    """
    VIZ-08: Time series — returns (line_fig, area_fig).
    Expects a long-form DataFrame with year, name, count columns.
    """
    # Top N by total
    totals = df.groupby(name_col)[count_col].sum().nlargest(top_n)
    top_names = totals.index.tolist()
    df_top = df[df[name_col].isin(top_names)].copy()
    df_other = df[~df[name_col].isin(top_names)].groupby(year_col)[count_col].sum().reset_index()
    df_other[name_col] = "Other"
    df_plot = pd.concat([df_top, df_other], ignore_index=True)

    palette = px.colors.qualitative.Plotly + px.colors.qualitative.Set2

    line_fig = px.line(
        df_plot, x=year_col, y=count_col, color=name_col,
        markers=True, color_discrete_sequence=palette,
    )
    line_fig.update_layout(
        template="plotly_white", height=height,
        xaxis=dict(dtick=1), margin=dict(t=30, l=50, r=30, b=50),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
    )

    # Stacked area (relative)
    df_pivot = df_plot.pivot_table(index=year_col, columns=name_col, values=count_col, fill_value=0)
    df_share = df_pivot.div(df_pivot.sum(axis=1), axis=0)
    df_share_long = df_share.reset_index().melt(id_vars=year_col, var_name=name_col, value_name="Share")

    area_fig = px.area(
        df_share_long, x=year_col, y="Share", color=name_col,
        color_discrete_sequence=palette,
    )
    area_fig.update_layout(
        template="plotly_white", height=height,
        xaxis=dict(dtick=1), yaxis=dict(tickformat=".0%"),
        margin=dict(t=30, l=50, r=30, b=50),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
    )
    return line_fig, area_fig


def plot_reciprocity_scatter(
    df: pd.DataFrame,
    domestic_country: str,
    short_name: str,
    element_name: str = "",
    height: int = 550,
) -> go.Figure:
    """VIZ-06: Reciprocity bubble scatter."""
    fig = px.scatter(
        df,
        x="share_partner",
        y="share_inst",
        size="partner_total",
        size_max=40,
        color="geo",
        color_discrete_map={domestic_country: "#4e79a7", "International": "#e15759"},
        hover_name="name",
    )
    max_val = max(
        df["share_partner"].max() if len(df) else 0.1,
        df["share_inst"].max() if len(df) else 0.1,
    ) * 1.05
    fig.add_shape(
        type="line", x0=0, y0=0, x1=max_val, y1=max_val,
        line=dict(color="gray", dash="dash"),
    )
    fig.update_layout(
        template="plotly_white",
        height=height,
        xaxis=dict(
            title=f"Share of partner's {element_name} output",
            tickformat=".0%",
        ),
        yaxis=dict(
            title=f"Share of {short_name}'s {element_name} output",
            tickformat=".0%",
        ),
        margin=dict(t=30, l=50, r=30, b=50),
    )
    return fig


def plot_pie_doctype(data: dict[str, int], height: int = 250) -> go.Figure:
    """VIZ-07: Document type donut / pie chart."""
    from lib.config import DOCTYPE_COLORS
    labels = list(data.keys())
    values = list(data.values())
    colors = [DOCTYPE_COLORS.get(l, "#999") for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=colors, line=dict(color="white", width=2)),
        textinfo="percent", textposition="inside", hole=0.0,
    ))
    fig.update_layout(
        height=height,
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5),
    )
    return fig


# =====================================================================
#  Table builders
# =====================================================================

def build_overview_table(
    df: pd.DataFrame,
    level: str,
    show_emoji: bool = True,
) -> pd.DataFrame:
    """
    Build a display-ready overview table from thematic_overview rows.
    Returns a DataFrame for st.dataframe.
    """
    from lib.config import CAGR_COL

    pct_col = pubs_pct_col()
    rows = []
    for _, r in df.iterrows():
        name = r.get("name", "")
        domain_id = safe_int(r.get("domain_id", 0))
        emoji = domain_emoji(domain_name_for_id(domain_id)) if show_emoji else ""
        row = {
            "": emoji,
            "Name": name,
            "Pubs": safe_int(r.get("pubs_total")),
            "% Total": round(safe_float(r.get(pct_col)) * 100, 1),
            "FWCI median": round(safe_float(r.get("fwci_median")), 2),
            "% Int'l": round(safe_float(r.get("pct_international")) * 100, 1),
            f"CAGR": format_cagr(r.get(CAGR_COL)),
        }
        rows.append(row)
    return pd.DataFrame(rows)


# =====================================================================
#  Misc
# =====================================================================

def darken_hex(hex_color: str, factor: float = 0.65) -> str:
    """Darken a hex color by a factor (0=black, 1=unchanged)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    r, g, b = int(r * factor), int(g * factor), int(b * factor)
    return f"#{r:02x}{g:02x}{b:02x}"
