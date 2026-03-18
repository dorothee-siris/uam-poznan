"""
Topic Explorer — Two-mode BERTopic landscape.

Mode 1 (Cluster): hover = cluster info, color = metric intensity per cluster,
                   search highlights % representation by cluster.
Mode 2 (Document): hover = publication info, color = binary highlight of
                    selected elements (up to 10, each a different color).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Topic Explorer", layout="wide")

# ── Cluster keywords (hardcoded from BERTopic output) ───────────────────────

CLUSTER_KEYWORDS = {
    0: "synthesis, complexes, ii, characterization, crystal, properties, new, structure, complex, derivatives",
    1: "explores, french, concept, focusing, context, political, century, social, role, relationship",
    2: "research, evolution, topic, formation, understand, investigates, basin, mantle, geological, earth",
    3: "antibiotic, stewardship, dental, antibiotics, care, oral, antimicrobial, dentistry, orthodontic, french",
    4: "stroke, thrombectomy, ischemic, endovascular, acute, occlusion, patients, artery, vessel, outcomes",
    5: "heart, failure, patients, ejection, fraction, kidney, cardiovascular, outcomes, disease, diabetes",
    6: "alloy, alloys, steel, high, behavior, ni, phase, mechanical, mn, microstructure",
    7: "molecular, protein, using, quantum, electron, dynamics, binding, topic, molecules, skin",
    8: "osteoarthritis, knee, sclerosis, hip, patients, arthritis, flare, multiple, disease, alzheimer",
    9: "using, equations, developing, quantum, stochastic, data, equation, topic, research, systems",
    10: "composites, flame, reinforced, printing, epoxy, properties, 3d, nanocomposites, mechanical, polymer",
    11: "research, topic, investigates, respiratory, surgery, children, study, patients, asthma, syndrome",
    12: "dc, energy, control, power, converter, wind, synchronous, voltage, magnet, permanent",
    13: "systems, nonlinear, control, linear, observer, time, design, estimation, output, varying",
    14: "plasma, flow, using, fluid, topic, media, porous, heat, turbulence, research",
    15: "research, soil, forest, plant, species, investigates, topic, climate, tree, forests",
    16: "wood, lignin, adhesives, tannin, formaldehyde, adhesive, using, steam, properties, torrefaction",
    17: "text, knowledge, language, learning, networks, social, developing, embeddings, natural, using",
    18: "education, students, teaching, learning, training, school, educational, language, skills, french",
    19: "innovation, impact, social, management, explores, financial, economic, research, relationship, market",
    20: "face, brain, sports, visual, human, epilepsy, physical, stimulation, activity, neural",
    21: "ulcerative, colitis, bowel, inflammatory, disease, patients, efficacy, safety, ibd, treatment",
    22: "cancer, cell, patients, cells, stem, crohn, disease, treatment, pet, efficacy",
    23: "materials, composites, wave, homogenization, damage, composite, using, acoustic, beams, numerical",
    24: "heat, fuel, solar, membrane, thermal, proton, exchange, performance, water, building",
    25: "research, using, topic, properties, films, spin, development, focuses, adsorption, magnetic",
    26: "bone, tissue, hydrogels, engineering, development, based, research, delivery, properties, applications",
    27: "rna, dna, sars, cov, role, integrative, modifications, conjugative, research, topic",
    28: "photodynamic, therapy, cancer, nanoparticles, drug, delivery, targeted, fluorescence, development, tumor",
    29: "developing, systems, manufacturing, maintenance, industry, scheduling, based, design, data, iot",
    30: "protein, essential, research, properties, antioxidant, milk, topic, powders, food, proteins",
    31: "health, care, disorder, adolescents, covid, 19, bipolar, explores, relationship, children",
}

# ── Color palettes ──────────────────────────────────────────────────────────

CLUSTER_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    "#393b79", "#5254a3", "#6b6ecf", "#9c9ede", "#637939",
    "#8ca252", "#b5cf6b", "#cedb9c", "#8c6d31", "#bd9e39",
    "#e7ba52", "#e7cb94",
]

HIGHLIGHT_PALETTE = [
    "#E63946", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0",
    "#00BCD4", "#FF5722", "#8BC34A", "#3F51B5", "#FFC107",
]

GREY = "#CCCCCC"

# ── Data loading ────────────────────────────────────────────────────────────


@st.cache_data
def load_data():
    df = pd.read_parquet("data/pubs_test.parquet")
    df["pub_year"] = pd.to_numeric(df["pub_year"], errors="coerce").astype("Int64")
    df["fwci"] = pd.to_numeric(df["fwci"], errors="coerce")
    df["tm_cluster"] = pd.to_numeric(df["tm_cluster"], errors="coerce").astype("Int64")
    df["tm_x"] = pd.to_numeric(df["tm_x"], errors="coerce")
    df["tm_y"] = pd.to_numeric(df["tm_y"], errors="coerce")
    for bc in ["is_oa", "is_international", "is_company"]:
        df[bc] = df[bc].astype(bool)
    for col in ["p_topic", "p_subfield", "p_field", "p_domain"]:
        df[f"{col}_label"] = df[col].astype(str).str.split(":", n=1).str[1].str.strip()
    df["title_short"] = df["title"].fillna("").str[:90]
    df["oa_label"] = np.where(df["is_oa"], "✅ OA", "❌ Non-OA")
    return df


@st.cache_data
def build_inst_lookup(_df):
    rows = _df[["openalex_id", "inst_name"]].dropna(subset=["inst_name"])
    rows = rows.assign(institution=rows["inst_name"].str.split("|")).explode("institution")
    rows["institution"] = rows["institution"].str.strip()
    rows = rows[rows["institution"] != ""]
    return rows[["openalex_id", "institution"]].drop_duplicates()


@st.cache_data
def compute_centroids(_df):
    valid = _df[_df["tm_cluster"] != -1]
    return (
        valid.groupby("tm_cluster")
        .agg(cx=("tm_x", "mean"), cy=("tm_y", "mean"), tm_label=("tm_label", "first"))
        .reset_index()
    )


@st.cache_data
def compute_cluster_stats(_df):
    return (
        _df.groupby("tm_cluster")
        .agg(
            n_pubs=("openalex_id", "size"),
            pct_oa=("is_oa", "mean"),
            pct_intl=("is_international", "mean"),
            pct_company=("is_company", "mean"),
            fwci_mean=("fwci", "mean"),
            fwci_median=("fwci", "median"),
            median_year=("pub_year", "median"),
        )
        .reset_index()
    )


# ── Helpers ─────────────────────────────────────────────────────────────────


def wrap_label(text, max_chars=28):
    words = text.split()
    lines, current = [], []
    for w in words:
        current.append(w)
        if len(" ".join(current)) > max_chars:
            lines.append(" ".join(current))
            current = []
    if current:
        lines.append(" ".join(current))
    return "<br>".join(lines)


def hex_to_rgba(hex_color, alpha=0.75):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def interpolate_color(value, vmin, vmax, low_rgb, high_rgb):
    if vmax == vmin:
        t = 0.5
    else:
        t = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    r = int(low_rgb[0] + t * (high_rgb[0] - low_rgb[0]))
    g = int(low_rgb[1] + t * (high_rgb[1] - low_rgb[1]))
    b = int(low_rgb[2] + t * (high_rgb[2] - low_rgb[2]))
    return f"#{r:02x}{g:02x}{b:02x}"


METRIC_RAMPS = {
    "% Open Access":   {"low": (230, 230, 230), "high": (33, 150, 243),  "scale": "Blues"},
    "% International": {"low": (230, 230, 230), "high": (25, 118, 210),  "scale": "Blues"},
    "% Industry":      {"low": (230, 230, 230), "high": (56, 142, 60),   "scale": "Greens"},
    "Mean FWCI":       {"low": (236, 135, 115), "high": (96, 204, 170),  "scale": "RdYlGn"},
    "Median FWCI":     {"low": (236, 135, 115), "high": (96, 204, 170),  "scale": "RdYlGn"},
    "Median year":     {"low": (68, 1, 84),     "high": (253, 231, 37),  "scale": "Viridis"},
    "_search":         {"low": (230, 230, 230), "high": (230, 57, 70),   "scale": "Reds"},
}


# ── Load & filter ───────────────────────────────────────────────────────────

df_raw = load_data()
inst_lookup = build_inst_lookup(df_raw)

st.sidebar.title("🗺️ Topic Explorer")

# Mode
mode = st.sidebar.radio("Mode", ["🔬 Cluster view", "📄 Document view"], horizontal=True)
is_cluster_mode = mode.startswith("🔬")

st.sidebar.markdown("---")

# Shared filters
with st.sidebar.expander("🎛️ Data filters", expanded=False):
    year_min, year_max = int(df_raw["pub_year"].min()), int(df_raw["pub_year"].max())
    year_range = st.slider("Year", year_min, year_max, (year_min, year_max))
    all_types = sorted(df_raw["pub_type"].dropna().unique().tolist())
    selected_types = st.multiselect("Document type", all_types, default=all_types)
    oa_filter = st.radio("Open Access", ["All", "OA only", "Non-OA only"], horizontal=True)
    intl_filter = st.radio("International collab", ["All", "International only", "Domestic only"], horizontal=True)
    industry_filter = st.radio("Industry collab", ["All", "With industry", "Without industry"], horizontal=True)

mask = (df_raw["pub_year"] >= year_range[0]) & (df_raw["pub_year"] <= year_range[1])
if selected_types:
    mask &= df_raw["pub_type"].isin(selected_types)
if oa_filter == "OA only":
    mask &= df_raw["is_oa"]
elif oa_filter == "Non-OA only":
    mask &= ~df_raw["is_oa"]
if intl_filter == "International only":
    mask &= df_raw["is_international"]
elif intl_filter == "Domestic only":
    mask &= ~df_raw["is_international"]
if industry_filter == "With industry":
    mask &= df_raw["is_company"]
elif industry_filter == "Without industry":
    mask &= ~df_raw["is_company"]

df = df_raw[mask].copy()
if len(df) < 50:
    st.warning("Filters are too restrictive — fewer than 50 publications remain.")

cluster_stats = compute_cluster_stats(df)
centroids = compute_centroids(df)
cstats = cluster_stats.set_index("tm_cluster")

# Merge cluster stats onto df for hover
df = df.merge(cluster_stats, on="tm_cluster", how="left", suffixes=("", "_cls"))
for c in ["n_pubs", "pct_oa", "pct_intl", "pct_company", "fwci_mean", "fwci_median", "median_year"]:
    df[c] = df[c].fillna(0)
df["keywords"] = df["tm_cluster"].map(CLUSTER_KEYWORDS).fillna("")

n_clusters_display = df.loc[df["tm_cluster"] != -1, "tm_cluster"].nunique()

show_labels = st.sidebar.toggle("Show cluster labels", value=True)


# ═══════════════════════════════════════════════════════════════════════════
# CLUSTER MODE
# ═══════════════════════════════════════════════════════════════════════════

if is_cluster_mode:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Cluster coloring")

    search_type = st.sidebar.selectbox(
        "🔍 Search by",
        ["None", "Partner institution", "OA Topic", "OA Subfield", "OA Field", "OA Domain"],
    )

    search_match_ids = set()

    if search_type == "Partner institution":
        sq = st.sidebar.text_input("Institution name (substring)", key="cl_inst_search")
        if sq.strip():
            matched = inst_lookup[inst_lookup["institution"].str.contains(sq.strip(), case=False, na=False)]
            if not matched.empty:
                search_match_ids = set(matched["openalex_id"])
                st.sidebar.caption(
                    f"{len(search_match_ids):,} publications across "
                    f"{matched['institution'].nunique()} institution(s)"
                )
            else:
                st.sidebar.caption("No institutions matched.")

    elif search_type != "None":
        col_map = {
            "OA Topic": "p_topic_label", "OA Subfield": "p_subfield_label",
            "OA Field": "p_field_label", "OA Domain": "p_domain_label",
        }
        tax_col = col_map[search_type]
        sq = st.sidebar.text_input(f"Search {search_type}", key="cl_tax_search")
        if sq.strip():
            all_vals = sorted(df[tax_col].dropna().unique())
            matched_vals = [v for v in all_vals if sq.strip().lower() in v.lower()]
            if matched_vals:
                sel = st.sidebar.selectbox(f"Select {search_type}", matched_vals, key="cl_tax_sel")
                search_match_ids = set(df.loc[df[tax_col] == sel, "openalex_id"])
                st.sidebar.caption(f"{len(search_match_ids):,} publications in '{sel}'")
            else:
                st.sidebar.caption("No match found.")

    search_active = len(search_match_ids) > 0

    metric_options = [
        "Cluster identity", "% Open Access", "% International", "% Industry",
        "Mean FWCI", "Median FWCI", "Median year",
    ]
    if search_active:
        st.sidebar.selectbox("🎨 Color by metric", metric_options, disabled=True,
                             help="Overridden by active search", key="cl_met_dis")
        color_metric = None
    else:
        color_metric = st.sidebar.selectbox("🎨 Color by metric", metric_options, key="cl_met")

    # ── Compute cluster colors ──────────────────────────────────────────────

    label_color_map = {}  # cluster_id → hex color (for label backgrounds)

    if search_active:
        df["_sm"] = df["openalex_id"].isin(search_match_ids)
        pct_by_cluster = df.groupby("tm_cluster")["_sm"].mean().to_dict()
        df.drop(columns=["_sm"], inplace=True)
        vmax = max(pct_by_cluster.values()) if pct_by_cluster else 1
        vmin = 0
        ramp = METRIC_RAMPS["_search"]
        for cid in df["tm_cluster"].unique():
            if cid == -1:
                label_color_map[cid] = GREY
            else:
                label_color_map[cid] = interpolate_color(
                    pct_by_cluster.get(cid, 0), vmin, vmax, ramp["low"], ramp["high"]
                )
        color_array = df["tm_cluster"].map(label_color_map).values
        active_scale, active_cmin, active_cmax = ramp["scale"], vmin, vmax
        colorbar_title = "% match"

    elif color_metric == "Cluster identity":
        for i, cid in enumerate(sorted(df["tm_cluster"].unique())):
            label_color_map[cid] = GREY if cid == -1 else CLUSTER_PALETTE[int(cid) % len(CLUSTER_PALETTE)]
        color_array = df["tm_cluster"].map(label_color_map).values
        active_scale = active_cmin = active_cmax = colorbar_title = None

    else:
        stat_col_map = {
            "% Open Access": "pct_oa", "% International": "pct_intl",
            "% Industry": "pct_company", "Mean FWCI": "fwci_mean",
            "Median FWCI": "fwci_median", "Median year": "median_year",
        }
        stat_col = stat_col_map[color_metric]
        ramp = METRIC_RAMPS[color_metric]

        if "FWCI" in color_metric:
            vmin, vmax = 0, 3
        elif color_metric == "Median year":
            valid_y = cstats.loc[cstats.index != -1, stat_col].dropna()
            vmin = float(valid_y.min()) if len(valid_y) else 2020
            vmax = float(valid_y.max()) if len(valid_y) else 2024
        else:
            vmin, vmax = 0, 1

        for cid in df["tm_cluster"].unique():
            if cid == -1:
                label_color_map[cid] = GREY
            elif cid in cstats.index:
                v = cstats.loc[cid, stat_col]
                label_color_map[cid] = interpolate_color(
                    v if pd.notna(v) else vmin, vmin, vmax, ramp["low"], ramp["high"]
                )
            else:
                label_color_map[cid] = GREY

        color_array = df["tm_cluster"].map(label_color_map).values
        active_scale, active_cmin, active_cmax = ramp["scale"], vmin, vmax
        colorbar_title = color_metric

    # ── Cluster-mode hover texts ────────────────────────────────────────────

    hover_texts = []
    for _, r in df.iterrows():
        cid = r["tm_cluster"]
        if cid == -1:
            hover_texts.append("<b>Outlier (unassigned)</b>")
            continue
        kw = CLUSTER_KEYWORDS.get(int(cid), "")
        kw_short = kw[:80] + "…" if len(kw) > 80 else kw
        hover_texts.append(
            f"<b>{r['tm_label']}</b><br>"
            f"<i style='color:#666'>{kw_short}</i><br>"
            f"───────────────────<br>"
            f"Publications: {int(r['n_pubs']):,}<br>"
            f"Open Access: {r['pct_oa']:.0%}  ·  International: {r['pct_intl']:.0%}  ·  Industry: {r['pct_company']:.0%}<br>"
            f"Mean FWCI: {r['fwci_mean']:.2f}  ·  Median FWCI: {r['fwci_median']:.2f}<br>"
            f"Median year: {r['median_year']:.0f}"
        )

    # ── Build cluster-mode figure ───────────────────────────────────────────

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=df["tm_x"], y=df["tm_y"], mode="markers",
        marker=dict(size=5, opacity=0.55, color=color_array, line=dict(width=0)),
        text=hover_texts, hoverinfo="text", name="",
    ))

    # Colorbar via invisible trace
    if active_scale and active_cmin is not None:
        is_pct = colorbar_title in ("% Open Access", "% International", "% Industry", "% match")
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(
                size=0, color=[active_cmin], colorscale=active_scale,
                cmin=active_cmin, cmax=active_cmax,
                colorbar=dict(title=colorbar_title, thickness=15, len=0.6,
                              tickformat=".0%" if is_pct else ""),
                showscale=True,
            ),
            hoverinfo="skip", showlegend=False,
        ))


# ═══════════════════════════════════════════════════════════════════════════
# DOCUMENT MODE
# ═══════════════════════════════════════════════════════════════════════════

else:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Highlight elements")

    highlight_type = st.sidebar.selectbox(
        "Element type",
        ["Cluster", "Partner institution", "OA Topic", "OA Subfield",
         "OA Field", "OA Domain", "Open Access"],
        key="doc_hl_type",
    )

    selected_elements = []

    if highlight_type == "Cluster":
        cluster_names = sorted(df.loc[df["tm_cluster"] != -1, "tm_label"].unique().tolist())
        selected_elements = st.sidebar.multiselect(
            "Select clusters (max 10)", cluster_names, max_selections=10, key="doc_cl_sel"
        )

    elif highlight_type == "Partner institution":
        inst_search = st.sidebar.text_input("Search institution name", key="doc_inst_search")
        if inst_search.strip():
            matched = inst_lookup[
                inst_lookup["institution"].str.contains(inst_search.strip(), case=False, na=False)
            ]["institution"].unique()
            unique_insts = sorted(matched)[:100]
            if unique_insts:
                selected_elements = st.sidebar.multiselect(
                    "Select institution(s) (max 10)", unique_insts, max_selections=10, key="doc_inst_sel"
                )
            else:
                st.sidebar.caption("No institutions matched.")

    elif highlight_type == "Open Access":
        oa_choice = st.sidebar.radio("Show", ["OA publications", "Non-OA publications"], key="doc_oa_r")
        selected_elements = [oa_choice]

    else:
        col_map = {
            "OA Topic": "p_topic_label", "OA Subfield": "p_subfield_label",
            "OA Field": "p_field_label", "OA Domain": "p_domain_label",
        }
        tax_col = col_map[highlight_type]
        tax_search = st.sidebar.text_input(f"Search {highlight_type}", key="doc_tax_search")
        all_vals = sorted(df[tax_col].dropna().unique())
        if tax_search.strip():
            all_vals = [v for v in all_vals if tax_search.strip().lower() in v.lower()]
        if all_vals:
            selected_elements = st.sidebar.multiselect(
                f"Select (max 10)", all_vals[:200], max_selections=10, key="doc_tax_sel"
            )

    # ── Compute masks per element ───────────────────────────────────────────

    element_masks = {}

    if highlight_type == "Cluster":
        for elem in selected_elements:
            element_masks[elem] = (df["tm_label"] == elem).values
    elif highlight_type == "Partner institution":
        for elem in selected_elements:
            ids = set(inst_lookup.loc[inst_lookup["institution"] == elem, "openalex_id"])
            element_masks[elem] = df["openalex_id"].isin(ids).values
    elif highlight_type == "Open Access":
        if selected_elements:
            if selected_elements[0] == "OA publications":
                element_masks["OA"] = df["is_oa"].values
            else:
                element_masks["Non-OA"] = (~df["is_oa"]).values
    else:
        col_map = {
            "OA Topic": "p_topic_label", "OA Subfield": "p_subfield_label",
            "OA Field": "p_field_label", "OA Domain": "p_domain_label",
        }
        tax_col = col_map[highlight_type]
        for elem in selected_elements:
            element_masks[elem] = (df[tax_col] == elem).values

    has_hl = len(element_masks) > 0
    color_array = np.full(len(df), GREY, dtype=object)
    opacity_array = np.full(len(df), 0.12 if has_hl else 0.55)
    size_array = np.full(len(df), 3 if has_hl else 5)

    element_color_map = {}
    for i, (elem, m) in enumerate(element_masks.items()):
        c = HIGHLIGHT_PALETTE[i % len(HIGHLIGHT_PALETTE)]
        element_color_map[elem] = c
        color_array[m] = c
        opacity_array[m] = 0.85
        size_array[m] = 7

    # ── Document-mode hover ─────────────────────────────────────────────────

    customdata = np.column_stack([
        df["title_short"].values, df["pub_year"].values, df["pub_type"].values,
        df["oa_label"].values, df["tm_label"].values, df["p_topic_label"].values,
        df["p_field_label"].values, df["p_domain_label"].values,
    ])

    hover_template = (
        "<b>%{customdata[0]}</b><br>"
        "───────────────────<br>"
        "Year: %{customdata[1]}  ·  Type: %{customdata[2]}  ·  %{customdata[3]}<br>"
        "Cluster: %{customdata[4]}<br>"
        "Topic: %{customdata[5]}<br>"
        "Field: %{customdata[6]}  ·  Domain: %{customdata[7]}"
        "<extra></extra>"
    )

    # ── Build document-mode figure ──────────────────────────────────────────

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=df["tm_x"], y=df["tm_y"], mode="markers",
        marker=dict(size=size_array, opacity=opacity_array, color=color_array, line=dict(width=0)),
        customdata=customdata, hovertemplate=hover_template, name="",
    ))

    # Legend in sidebar
    if element_color_map:
        st.sidebar.markdown("**Highlighted:**")
        for elem, c in element_color_map.items():
            n = int(element_masks[elem].sum())
            st.sidebar.markdown(
                f'<span style="color:{c}; font-size:18px;">●</span> {elem} ({n:,})',
                unsafe_allow_html=True,
            )

    label_color_map = None  # document mode uses neutral labels


# ═══════════════════════════════════════════════════════════════════════════
# SHARED: Cluster label annotations + layout
# ═══════════════════════════════════════════════════════════════════════════

if show_labels and not centroids.empty:
    for _, crow in centroids.iterrows():
        cid = int(crow["tm_cluster"])
        wrapped = wrap_label(crow["tm_label"])

        if is_cluster_mode and label_color_map and cid in label_color_map:
            bg = hex_to_rgba(label_color_map[cid], 0.78)
        else:
            bg = "rgba(255,255,255,0.72)"

        fig.add_annotation(
            x=crow["cx"], y=crow["cy"],
            text=f"<b>{wrapped}</b>",
            showarrow=False,
            font=dict(size=11, color="#1a1a1a"),
            bgcolor=bg,
            bordercolor="rgba(150,150,150,0.4)",
            borderwidth=1,
            borderpad=5,
            opacity=0.92,
        )

mode_tag = "Cluster" if is_cluster_mode else "Document"
fig.update_layout(
    title=dict(
        text=f"Topic Landscape ({mode_tag} view) — {len(df):,} pubs · {n_clusters_display} clusters",
        font=dict(size=16),
    ),
    height=850,
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(visible=False),
    yaxis=dict(visible=False, scaleanchor="x"),
    showlegend=False,
    uirevision="constant",
    margin=dict(t=50, l=10, r=10, b=10),
)

# ── Render ──────────────────────────────────────────────────────────────────

st.title("🗺️ Topic Explorer")

event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="topic_map")

selected_points = event.selection.points if event and event.selection else []
if selected_points:
    try:
        idx = selected_points[0]["point_index"]
        row = df.iloc[idx]
        url = f"https://openalex.org/works/{row['openalex_id']}"
        st.markdown(
            f"**Selected:** [{row['title'][:100]}]({url})  "
            f"· {row['pub_year']} · {row['pub_type']} · Cluster: *{row['tm_label']}*  \n"
            f"🔗 [Open in OpenAlex]({url})"
        )
    except (IndexError, KeyError):
        pass
else:
    st.info("Click a dot on the map to see publication details and an OpenAlex link.")
