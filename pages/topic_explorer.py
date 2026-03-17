"""
Topic Explorer — BERTopic cluster landscape with interactive filters.

All logic is self-contained in this file for now.
Move helpers to lib/ modules when stabilised.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Topic Explorer", layout="wide")

# ── Data loading ────────────────────────────────────────────────────────────


@st.cache_data
def load_topic_model():
    df = pd.read_parquet("data/pubs_test.parquet")

    # Parse taxonomy labels
    for col in ["p_topic", "p_subfield", "p_field", "p_domain"]:
        label_col = f"{col}_label"
        df[label_col] = df[col].astype(str).str.split(":", n=1).str[1].str.strip()

    # Truncated title for hover
    df["title_short"] = df["title"].fillna("").str[:90]

    # OA display label
    df["oa_label"] = np.where(df["is_oa"], "✅ OA", "❌ Non-OA")

    return df


@st.cache_data
def build_inst_lookup(_df):
    """Exploded institution lookup for partner filtering."""
    rows = _df[["openalex_id", "inst_name"]].dropna(subset=["inst_name"])
    rows = rows.assign(institution=rows["inst_name"].str.split("|")).explode("institution")
    rows["institution"] = rows["institution"].str.strip()
    rows = rows[rows["institution"] != ""]
    return rows[["openalex_id", "institution"]].drop_duplicates()


@st.cache_data
def compute_centroids(_df):
    """Cluster centroids for label annotations (excluding outliers)."""
    valid = _df[_df["tm_cluster"] != -1]
    centroids = (
        valid.groupby("tm_cluster")
        .agg(tm_x=("tm_x", "mean"), tm_y=("tm_y", "mean"), tm_label=("tm_label", "first"))
        .reset_index()
    )
    return centroids


@st.cache_data
def compute_cluster_stats(_df):
    """Per-cluster aggregated metrics from the filtered DataFrame."""
    stats = (
        _df.groupby("tm_cluster")
        .agg(
            n_cluster=("openalex_id", "size"),
            median_fwci_cluster=("fwci", "median"),
            median_year_cluster=("pub_year", "median"),
            pct_intl_cluster=("is_international", "mean"),
            pct_company_cluster=("is_company", "mean"),
            pct_oa_cluster=("is_oa", "mean"),
        )
        .reset_index()
    )
    return stats


# ── Load data ───────────────────────────────────────────────────────────────

df_raw = load_topic_model()
inst_lookup = build_inst_lookup(df_raw)
centroids_full = compute_centroids(df_raw)

# ── Sidebar ─────────────────────────────────────────────────────────────────

st.sidebar.title("🗺️ Topic Explorer")
st.sidebar.markdown("---")

# 1. Topic search
topic_search = st.sidebar.text_input("🔍 Search topic", "", help="Substring match on cluster labels")

# 2. Partner institution filter
partner_text = st.sidebar.text_input("🏛 Filter by partner institution", "", help="Search for a collaborating institution")
selected_partners = []
if partner_text.strip():
    matches = inst_lookup[inst_lookup["institution"].str.contains(partner_text.strip(), case=False, na=False)]
    unique_insts = sorted(matches["institution"].unique())[:50]  # cap suggestions
    if unique_insts:
        selected_partners = st.sidebar.multiselect("Select institution(s)", unique_insts, default=[])
    else:
        st.sidebar.caption("No institutions matched.")

# Determine if a highlight mode is active
highlight_active = bool(topic_search.strip()) or bool(selected_partners)

# 3. Color mode
color_options = [
    "Topic (categorical)",
    "Open Access",
    "Median FWCI",
    "Median year",
    "% International",
    "% Industry",
    "Domain",
    "Field",
]

if highlight_active:
    st.sidebar.selectbox(
        "🎨 Color dots by",
        color_options,
        disabled=True,
        help="Disabled while search/partner filter is active",
        key="color_mode_disabled",
    )
    color_mode = None  # signal: highlight mode overrides
else:
    color_mode = st.sidebar.selectbox("🎨 Color dots by", color_options, key="color_mode")

# OA sub-toggle
oa_view = "Individual dots"
if color_mode == "Open Access":
    oa_view = st.sidebar.radio("OA view", ["Individual dots", "Cluster intensity"], horizontal=True)

# 4. Cluster labels
show_labels = st.sidebar.toggle("Show cluster labels", value=True)

# 5. Metadata filters
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

year_min, year_max = int(df_raw["pub_year"].min()), int(df_raw["pub_year"].max())
year_range = st.sidebar.slider("Year", year_min, year_max, (year_min, year_max))

all_types = sorted(df_raw["pub_type"].dropna().unique().tolist())
selected_types = st.sidebar.multiselect("Document type", all_types, default=all_types)

oa_filter = st.sidebar.radio("Open Access", ["All", "OA only", "Non-OA only"], horizontal=True)
intl_filter = st.sidebar.radio("International collab", ["All", "International only", "Domestic only"], horizontal=True)
industry_filter = st.sidebar.radio("Industry collab", ["All", "With industry", "Without industry"], horizontal=True)

# ── Apply filters ───────────────────────────────────────────────────────────


@st.cache_data
def apply_filters(_df, yr, types, oa, intl, industry):
    mask = (_df["pub_year"] >= yr[0]) & (_df["pub_year"] <= yr[1])
    if types:
        mask &= _df["pub_type"].isin(types)
    if oa == "OA only":
        mask &= _df["is_oa"]
    elif oa == "Non-OA only":
        mask &= ~_df["is_oa"]
    if intl == "International only":
        mask &= _df["is_international"]
    elif intl == "Domestic only":
        mask &= ~_df["is_international"]
    if industry == "With industry":
        mask &= _df["is_company"]
    elif industry == "Without industry":
        mask &= ~_df["is_company"]
    return _df[mask].copy()


df = apply_filters(df_raw, year_range, selected_types, oa_filter, intl_filter, industry_filter)

if len(df) < 50:
    st.warning("Filters are too restrictive — fewer than 50 publications remain.")

# ── Compute cluster stats on filtered data ──────────────────────────────────

cluster_stats = compute_cluster_stats(df)
df = df.merge(cluster_stats, on="tm_cluster", how="left")

# Fill NaN stats for display
for c in ["n_cluster", "median_fwci_cluster", "median_year_cluster", "pct_intl_cluster", "pct_company_cluster"]:
    df[c] = df[c].fillna(0)

# ── Determine highlight masks ───────────────────────────────────────────────

highlight_mask = pd.Series(True, index=df.index)

if topic_search.strip():
    topic_match = df["tm_label"].str.contains(topic_search.strip(), case=False, na=False)
    n_topics = df.loc[topic_match, "tm_cluster"].nunique()
    n_pubs = topic_match.sum()
    st.sidebar.caption(f"{n_topics} topics matched — {n_pubs:,} publications highlighted")
    highlight_mask &= topic_match

if selected_partners:
    partner_ids = set(
        inst_lookup[inst_lookup["institution"].isin(selected_partners)]["openalex_id"]
    )
    partner_match = df["openalex_id"].isin(partner_ids)
    st.sidebar.caption(f"{partner_match.sum():,} publications with selected partner(s)")
    highlight_mask &= partner_match

# ── Build color arrays ──────────────────────────────────────────────────────

GREY = "#CCCCCC"


def _build_highlight_arrays(df, mask):
    """Return color, opacity, size arrays for highlight mode."""
    colors = pd.Series(GREY, index=df.index)
    colors[mask] = "#E63946"
    opacity = pd.Series(0.25, index=df.index)
    opacity[mask] = 1.0
    sizes = pd.Series(4, index=df.index)
    sizes[mask] = 7
    return colors.values, opacity.values, sizes.values


def _cluster_color_continuous(df, metric_col, colorscale, cmin, cmax):
    """Map a per-cluster continuous metric to each dot, forcing outliers to grey."""
    arr = df[metric_col].values.copy().astype(float)
    is_outlier = (df["tm_cluster"] == -1).values
    # We'll use colorscale on the fig and override outliers via a separate trace
    return arr, is_outlier


# ── Assemble figure ─────────────────────────────────────────────────────────

n_clusters_display = df["tm_cluster"].nunique() - (1 if -1 in df["tm_cluster"].values else 0)

fig = go.Figure()

if highlight_active:
    # Highlight mode
    colors, opacities, sizes = _build_highlight_arrays(df, highlight_mask)
    fig.add_trace(
        go.Scattergl(
            x=df["tm_x"],
            y=df["tm_y"],
            mode="markers",
            marker=dict(size=sizes, color=colors, opacity=opacities, line=dict(width=0)),
            customdata=np.column_stack(
                [
                    df["title_short"],
                    df["pub_year"],
                    df["pub_type"],
                    df["oa_label"],
                    df["tm_label"],
                    df["n_cluster"],
                    df["median_fwci_cluster"],
                    df["median_year_cluster"],
                    df["pct_intl_cluster"],
                    df["pct_company_cluster"],
                ]
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "─────────────────────────<br>"
                "Year: %{customdata[1]}  ·  Type: %{customdata[2]}  ·  OA: %{customdata[3]}<br><br>"
                "<b>Cluster: %{customdata[4]}</b><br>"
                "Publications in cluster: %{customdata[5]}<br>"
                "Median FWCI: %{customdata[6]}<br>"
                "Median year: %{customdata[7]}<br>"
                "International: %{customdata[8]}  ·  Industry: %{customdata[9]}<br>"
                "<extra></extra>"
            ),
            name="",
        )
    )

elif color_mode == "Topic (categorical)":
    # Assign colors per cluster using Light24
    palette = [
        "#FD3216", "#00FE35", "#6A76FC", "#FED4C4", "#FE00CE",
        "#0DF9FF", "#F6F926", "#FF9616", "#DEA0FD", "#922B21",
        "#00B5F7", "#E2E2E2", "#C075A6", "#FC6955", "#3283FE",
        "#B00068", "#FF9408", "#14E1FF", "#D626FF", "#862A16",
        "#A777F1", "#87C55F", "#9CDED6", "#F6222E",
    ]
    clusters_unique = sorted(df["tm_cluster"].unique())
    cluster_color_map = {}
    ci = 0
    for cl in clusters_unique:
        if cl == -1:
            cluster_color_map[cl] = GREY
        else:
            cluster_color_map[cl] = palette[ci % len(palette)]
            ci += 1
    colors = df["tm_cluster"].map(cluster_color_map).values

    fig.add_trace(
        go.Scattergl(
            x=df["tm_x"],
            y=df["tm_y"],
            mode="markers",
            marker=dict(size=5, opacity=0.65, color=colors, line=dict(width=0)),
            customdata=np.column_stack(
                [
                    df["title_short"], df["pub_year"], df["pub_type"], df["oa_label"],
                    df["tm_label"], df["n_cluster"], df["median_fwci_cluster"],
                    df["median_year_cluster"], df["pct_intl_cluster"], df["pct_company_cluster"],
                ]
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "─────────────────────────<br>"
                "Year: %{customdata[1]}  ·  Type: %{customdata[2]}  ·  OA: %{customdata[3]}<br><br>"
                "<b>Cluster: %{customdata[4]}</b><br>"
                "Publications in cluster: %{customdata[5]}<br>"
                "Median FWCI: %{customdata[6]}<br>"
                "Median year: %{customdata[7]}<br>"
                "International: %{customdata[8]}  ·  Industry: %{customdata[9]}<br>"
                "<extra></extra>"
            ),
            name="",
        )
    )

elif color_mode == "Open Access" and oa_view == "Individual dots":
    colors = np.where(df["is_oa"], "#2196F3", GREY)
    fig.add_trace(
        go.Scattergl(
            x=df["tm_x"], y=df["tm_y"], mode="markers",
            marker=dict(size=5, opacity=0.65, color=colors, line=dict(width=0)),
            customdata=np.column_stack([
                df["title_short"], df["pub_year"], df["pub_type"], df["oa_label"],
                df["tm_label"], df["n_cluster"], df["median_fwci_cluster"],
                df["median_year_cluster"], df["pct_intl_cluster"], df["pct_company_cluster"],
            ]),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "─────────────────────────<br>"
                "Year: %{customdata[1]}  ·  Type: %{customdata[2]}  ·  OA: %{customdata[3]}<br><br>"
                "<b>Cluster: %{customdata[4]}</b><br>"
                "Publications in cluster: %{customdata[5]}<br>"
                "Median FWCI: %{customdata[6]}<br>"
                "Median year: %{customdata[7]}<br>"
                "International: %{customdata[8]}  ·  Industry: %{customdata[9]}<br>"
                "<extra></extra>"
            ),
            name="",
        )
    )
    st.sidebar.markdown("🔵 OA &nbsp;&nbsp; ⚪ Non-OA")

else:
    # Continuous per-cluster modes (+ OA cluster intensity, Domain, Field)
    # Build the metric column to color by
    needs_colorbar = True

    if color_mode == "Open Access":  # cluster intensity
        metric_col = "pct_oa_cluster"
        cscale, cmin, cmax = "Blues", 0, 1
        cbar_title = "Share OA"
    elif color_mode == "Median FWCI":
        metric_col = "median_fwci_cluster"
        cscale, cmin, cmax = "RdBu", 0, 3
        cbar_title = "Median FWCI"
    elif color_mode == "Median year":
        metric_col = "median_year_cluster"
        cscale = "Viridis"
        cmin = df["median_year_cluster"].min()
        cmax = df["median_year_cluster"].max()
        cbar_title = "Median year"
    elif color_mode == "% International":
        metric_col = "pct_intl_cluster"
        cscale, cmin, cmax = "Blues", 0, 1
        cbar_title = "% International"
    elif color_mode == "% Industry":
        metric_col = "pct_company_cluster"
        cscale, cmin, cmax = "Greens", 0, 1
        cbar_title = "% Industry"
    elif color_mode == "Domain":
        metric_col = "p_domain_label"
        needs_colorbar = False
        cscale = cmin = cmax = cbar_title = None
    elif color_mode == "Field":
        metric_col = "p_field_label"
        needs_colorbar = False
        cscale = cmin = cmax = cbar_title = None
    else:
        metric_col = None
        needs_colorbar = False
        cscale = cmin = cmax = cbar_title = None

    if metric_col and needs_colorbar:
        # Continuous per-cluster: split outlier cluster vs rest
        is_outlier = df["tm_cluster"] == -1
        df_main = df[~is_outlier]
        df_outlier = df[is_outlier]

        if not df_main.empty:
            color_vals = df_main[metric_col].values.astype(float)
            fig.add_trace(
                go.Scattergl(
                    x=df_main["tm_x"], y=df_main["tm_y"], mode="markers",
                    marker=dict(
                        size=5, opacity=0.65, color=color_vals,
                        colorscale=cscale, cmin=cmin, cmax=cmax,
                        colorbar=dict(title=cbar_title, thickness=15, len=0.6),
                        line=dict(width=0),
                    ),
                    customdata=np.column_stack([
                        df_main["title_short"], df_main["pub_year"], df_main["pub_type"],
                        df_main["oa_label"], df_main["tm_label"], df_main["n_cluster"],
                        df_main["median_fwci_cluster"], df_main["median_year_cluster"],
                        df_main["pct_intl_cluster"], df_main["pct_company_cluster"],
                    ]),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "─────────────────────────<br>"
                        "Year: %{customdata[1]}  ·  Type: %{customdata[2]}  ·  OA: %{customdata[3]}<br><br>"
                        "<b>Cluster: %{customdata[4]}</b><br>"
                        "Publications in cluster: %{customdata[5]}<br>"
                        "Median FWCI: %{customdata[6]}<br>"
                        "Median year: %{customdata[7]}<br>"
                        "International: %{customdata[8]}  ·  Industry: %{customdata[9]}<br>"
                        "<extra></extra>"
                    ),
                    name="",
                )
            )

        if not df_outlier.empty:
            fig.add_trace(
                go.Scattergl(
                    x=df_outlier["tm_x"], y=df_outlier["tm_y"], mode="markers",
                    marker=dict(size=4, opacity=0.3, color=GREY, line=dict(width=0)),
                    customdata=np.column_stack([
                        df_outlier["title_short"], df_outlier["pub_year"], df_outlier["pub_type"],
                        df_outlier["oa_label"], df_outlier["tm_label"], df_outlier["n_cluster"],
                        df_outlier["median_fwci_cluster"], df_outlier["median_year_cluster"],
                        df_outlier["pct_intl_cluster"], df_outlier["pct_company_cluster"],
                    ]),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "─────────────────────────<br>"
                        "Year: %{customdata[1]}  ·  Type: %{customdata[2]}  ·  OA: %{customdata[3]}<br><br>"
                        "<b>Cluster: %{customdata[4]} (outlier)</b><br>"
                        "<extra></extra>"
                    ),
                    name="outliers",
                )
            )

    elif metric_col and not needs_colorbar:
        # Categorical per-dot: Domain or Field
        categories = sorted(df[metric_col].dropna().unique())
        palette_name = "Pastel" if color_mode == "Domain" else "Alphabet"
        import plotly.express as px
        palette = getattr(px.colors.qualitative, palette_name)

        cat_color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}
        # Outliers still grey
        is_outlier = df["tm_cluster"] == -1
        colors = df[metric_col].map(cat_color_map).fillna(GREY).values
        colors = np.where(is_outlier, GREY, colors)

        fig.add_trace(
            go.Scattergl(
                x=df["tm_x"], y=df["tm_y"], mode="markers",
                marker=dict(size=5, opacity=0.65, color=colors, line=dict(width=0)),
                customdata=np.column_stack([
                    df["title_short"], df["pub_year"], df["pub_type"], df["oa_label"],
                    df["tm_label"], df["n_cluster"], df["median_fwci_cluster"],
                    df["median_year_cluster"], df["pct_intl_cluster"], df["pct_company_cluster"],
                ]),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "─────────────────────────<br>"
                    "Year: %{customdata[1]}  ·  Type: %{customdata[2]}  ·  OA: %{customdata[3]}<br><br>"
                    "<b>Cluster: %{customdata[4]}</b><br>"
                    "Publications in cluster: %{customdata[5]}<br>"
                    "Median FWCI: %{customdata[6]}<br>"
                    "Median year: %{customdata[7]}<br>"
                    "International: %{customdata[8]}  ·  Industry: %{customdata[9]}<br>"
                    "<extra></extra>"
                ),
                name="",
            )
        )

        # Caption legend for categorical
        legend_items = " &nbsp;·&nbsp; ".join(
            [f'<span style="color:{cat_color_map[c]}">●</span> {c}' for c in categories]
        )
        st.caption(f"**Legend:** {legend_items}", unsafe_allow_html=True)

# ── Cluster label annotations ───────────────────────────────────────────────

if show_labels:
    # Recompute centroids on filtered data
    centroids = compute_centroids(df)
    if not centroids.empty:
        fig.add_trace(
            go.Scatter(
                x=centroids["tm_x"],
                y=centroids["tm_y"],
                mode="text",
                text=centroids["tm_label"],
                textfont=dict(size=11, color="#333333"),
                hoverinfo="skip",
                name="labels",
            )
        )

# ── Layout ──────────────────────────────────────────────────────────────────

fig.update_layout(
    title=dict(
        text=f"Topic Landscape — {len(df):,} publications · {n_clusters_display} clusters",
        font=dict(size=16),
    ),
    height=800,
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    showlegend=False,
    uirevision="constant",
    margin=dict(t=50, l=10, r=10, b=10),
)

# ── Render chart with click events ──────────────────────────────────────────

st.title("🗺️ Topic Explorer")

event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="topic_map")

# ── Click-to-OpenAlex link panel ────────────────────────────────────────────

selected_points = event.selection.points if event and event.selection else []

if selected_points:
    try:
        pt = selected_points[0]
        idx = pt["point_index"]
        trace_idx = pt.get("curve_number", 0)

        # If we split main/outlier traces (continuous modes), adjust index
        if highlight_active or color_mode in ("Topic (categorical)", "Open Access", "Domain", "Field"):
            row = df.iloc[idx]
        else:
            # Continuous mode: trace 0 = non-outlier, trace 1 = outlier
            is_outlier = df["tm_cluster"] == -1
            if trace_idx == 0:
                row = df[~is_outlier].iloc[idx]
            else:
                row = df[is_outlier].iloc[idx]

        oa_id = row["openalex_id"]
        url = f"https://openalex.org/works/{oa_id}"
        st.markdown(
            f"**Selected:** [{row['title'][:100]}]({url})  "
            f"· {row['pub_year']} · {row['pub_type']} · Cluster: *{row['tm_label']}*  \n"
            f"🔗 [Open in OpenAlex]({url})"
        )
    except (IndexError, KeyError):
        pass
else:
    st.info("Click a dot on the map to see publication details and an OpenAlex link.")
