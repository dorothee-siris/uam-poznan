# pages/4___Thematic_Drilldown.py
# Thematic Drilldown — deep dive into any thematic element

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Thematic Drilldown — UAM", page_icon="🏛️", layout="wide")

from lib.config import (
    SHORT_NAME,
    SHORT_CODE,
    CAGR_COL,
    CAGR_LABEL,
    DOMESTIC_COUNTRY,
    DOMESTIC_LABEL,
    DOMESTIC_CODE,
    DOMAIN_COLORS,
    DOMAIN_NAMES_ORDERED,
    DOMAIN_ORDER,
    INITIATIVE_COL,
    INITIATIVE_LABEL,
    HAS_SI,
    SI_NATIONAL_COL,
    SI_NATIONAL_LABEL,
    HAS_NCI,
    HAS_DOMINANCE,
    HAS_PP_TOP,
    OA_TOPIC_LEVEL_NAME,
    PUBS_PCT_COL,
    DOMESTIC_PARTNER_COL_KEYWORD,
    YEARS,
)
from lib.data_cache import (
    get_overview_for_level,
    get_sublevels_for_parent,
    get_partners_for_element,
    get_authors_for_element,
    load_all_topics,
)
from lib.helpers import (
    build_overview_table,
    domain_color,
    domain_emoji,
    domain_name_for_id,
    format_cagr,
    format_fwci,
    format_int,
    format_pct,
    parse_authors,
    parse_domestic_partners,
    parse_fwci_boxplot,
    parse_int_partners,
    parse_reciprocity_partners,
    parse_year_counts,
    plot_horizontal_bar,
    plot_precomputed_boxplot,
    plot_reciprocity_scatter,
    plot_time_series,
    pubs_pct_col,
    render_domain_legend,
    safe_float,
    safe_int,
)


# =====================================================================
#  Sidebar — Element selector
# =====================================================================

st.sidebar.title("🔎 Thematic Drilldown")

# Level selector
LEVEL_OPTIONS = {
    "Domain": "domain",
    "Field": "field",
    "Subfield": "subfield",
    "Topic (OA)": OA_TOPIC_LEVEL_NAME,
}
level_label = st.sidebar.selectbox("Classification level", list(LEVEL_OPTIONS.keys()))
level = LEVEL_OPTIONS[level_label]

# Load elements for the chosen level
df_level = get_overview_for_level(level)
if df_level.empty:
    st.warning(f"No data available for level: {level_label}.")
    st.stop()

# Build selector with pub counts
df_level = df_level.sort_values("pubs_total", ascending=False)
options_map = {}
for _, r in df_level.iterrows():
    dom_name = domain_name_for_id(r.get("domain_id", 0))
    emoji = domain_emoji(dom_name)
    label = f"{emoji} {r['name']}  ({safe_int(r['pubs_total']):,} pubs)"
    options_map[label] = r
selected_label = st.sidebar.selectbox("Select element", list(options_map.keys()))
row = options_map[selected_label]

element_id = str(row["id"])
element_name = row["name"]
element_level = level

# Parent info
parent_name = row.get("parent_name", "")
domain_id = safe_int(row.get("domain_id", 0))
dom_display = domain_name_for_id(domain_id)

# ── Main header ───────────────────────────────────────────────────────

st.title(f"{domain_emoji(dom_display)} {element_name}")
if parent_name:
    st.caption(f"{level_label} in **{parent_name}** · {dom_display}")
else:
    st.caption(f"{level_label} · {dom_display}")
st.markdown(render_domain_legend(), unsafe_allow_html=True)
st.divider()


# =====================================================================
#  1. KPI Metrics  (TBL-01)
# =====================================================================

st.header("Key Indicators")

pct_col = pubs_pct_col()

# ── Row 1: Volume & Growth ───────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Publications", format_int(row.get("pubs_total")))
c2.metric(f"% of {SHORT_NAME} Total", format_pct(row.get(pct_col)))
c3.metric(f"CAGR {CAGR_LABEL}", format_cagr(row.get(CAGR_COL)))
c4.metric("FWCI (median)", format_fwci(row.get("fwci_median")))

# ── Row 2: Citation & Collaboration ──────────────────────────────────
row2_metrics = []
fwci_mean = row.get("fwci_mean")
if fwci_mean is not None and not (isinstance(fwci_mean, float) and pd.isna(fwci_mean)):
    row2_metrics.append(("FWCI (mean)", format_fwci(fwci_mean)))

pct_int = row.get("pct_international")
if pct_int is not None and not (isinstance(pct_int, float) and pd.isna(pct_int)):
    row2_metrics.append(("% International", format_pct(pct_int)))

pct_company = row.get("pct_company")
if pct_company is not None and not (isinstance(pct_company, float) and pd.isna(pct_company)):
    row2_metrics.append(("% Company", format_pct(pct_company)))

# pct_top10 and pct_top1 — only show if they exist in the data
for col_name, label in [("pct_top10", "% Top 10%"), ("pct_top1", "% Top 1%")]:
    val = row.get(col_name)
    if val is not None and not (isinstance(val, float) and pd.isna(val)):
        row2_metrics.append((label, format_pct(val)))

if row2_metrics:
    cols = st.columns(len(row2_metrics))
    for col, (lbl, val) in zip(cols, row2_metrics):
        col.metric(lbl, val)

# ── Row 3: Optional advanced metrics ─────────────────────────────────
optional_cols = []
if INITIATIVE_COL and row.get(INITIATIVE_COL) is not None:
    init_val = row.get(INITIATIVE_COL)
    if not (isinstance(init_val, float) and pd.isna(init_val)):
        optional_cols.append((f"% {INITIATIVE_LABEL}", format_pct(init_val)))

if HAS_SI and SI_NATIONAL_COL:
    si_val = row.get(SI_NATIONAL_COL)
    if si_val is not None and not (isinstance(si_val, float) and pd.isna(si_val)):
        optional_cols.append((SI_NATIONAL_LABEL, f"{safe_float(si_val):.2f}"))

if HAS_NCI:
    nci_val = row.get("nci")
    if nci_val is not None and not (isinstance(nci_val, float) and pd.isna(nci_val)):
        optional_cols.append(("NCI", f"{safe_float(nci_val):.2f}"))

if optional_cols:
    opt_columns = st.columns(len(optional_cols))
    for col, (lbl, val) in zip(opt_columns, optional_cols):
        col.metric(lbl, val)

st.divider()

# =====================================================================
#  2. FWCI Distribution + Publication Trend (side by side)
# =====================================================================

col_left, col_right = st.columns([1, 2])

with col_left:
    st.header("FWCI Distribution")
    bp_raw = row.get("fwci_boxplot", "")
    bp = parse_fwci_boxplot(bp_raw)
    if bp:
        show_ext = st.toggle("Include extreme values (p0, p100)", value=False, key="dd_ext")
        bp["category"] = element_name
        bp["color"] = domain_color(dom_display)
        bp["count"] = safe_int(row.get("pubs_total"))
        fig_bp = plot_precomputed_boxplot([bp], show_extremes=show_ext, height=350)
        fig_bp.update_layout(yaxis_title="FWCI")
        st.plotly_chart(fig_bp, use_container_width=True)
    else:
        st.info("No FWCI distribution data available.")

with col_right:
    st.header("Publication Trend")
    year_blob = row.get("pubs_per_year", "")
    year_counts = parse_year_counts(year_blob)
    if year_counts:
        df_years = pd.DataFrame(
            [{"Year": y, "Publications": c} for y, c in sorted(year_counts.items())]
        )
        fig_ts = go.Figure(go.Bar(
            x=df_years["Year"], y=df_years["Publications"],
            marker_color=domain_color(dom_display),
            text=df_years["Publications"],
            textposition="outside",
        ))
        fig_ts.update_layout(
            template="plotly_white",
            height=400,
            xaxis=dict(dtick=1, title="Year"),
            yaxis_title="Publications",
            margin=dict(t=30, l=50, r=30, b=50),
        )
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info("No yearly publication data available.")

st.divider()

# =====================================================================
#  4. Sublevels  (child breakdown)
# =====================================================================

# Determine child level name
CHILD_LEVELS = {
    "domain": ("field", "Fields"),
    "field": ("subfield", "Subfields"),
    "subfield": (OA_TOPIC_LEVEL_NAME, "Topics"),
}

if element_level in CHILD_LEVELS:
    child_level, child_label = CHILD_LEVELS[element_level]
    st.header(f"{child_label} within {element_name}")

    df_children = get_sublevels_for_parent(element_level, element_id)

    if not df_children.empty:
        # ── Top children bar chart ────────────────────────────────
        df_bar = df_children.nlargest(20, "pubs_total").copy()
        df_bar["pct_label"] = df_bar["pubs_pct_of_parent"].apply(
            lambda x: f"{safe_float(x) * 100:.1f}%"
        )
        df_bar["color"] = domain_color(dom_display)

        fig_bar = plot_horizontal_bar(
            df_bar,
            y_col="child_name",
            x_col="pubs_total",
            text_col="pct_label",
            color_col="color",
            xaxis_title="Publications",
            title=f"Top {min(20, len(df_bar))} {child_label}",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Time evolution of sub-levels ──────────────────────────
        st.subheader(f"Time Evolution of {child_label}")
        ts_rows = []
        for _, cr in df_children.iterrows():
            yc = parse_year_counts(cr.get("pubs_per_year", ""))
            for y, c in yc.items():
                ts_rows.append({"Year": y, "Name": cr["child_name"], "Count": c})

        if ts_rows:
            df_ts = pd.DataFrame(ts_rows)
            tab_abs, tab_rel = st.tabs(["Absolute", "Relative (%)"])
            line_fig, area_fig = plot_time_series(df_ts, top_n=10)
            with tab_abs:
                st.plotly_chart(line_fig, use_container_width=True)
            with tab_rel:
                st.plotly_chart(area_fig, use_container_width=True)
        else:
            st.info("No time-series data for sub-levels.")

        # ── Full table ────────────────────────────────────────────
        st.subheader(f"All {child_label}")
        tbl_children = []
        for _, cr in df_children.iterrows():
            tbl_children.append({
                "Name": cr.get("child_name", ""),
                "Pubs": safe_int(cr.get("pubs_total")),
                "% of Parent": round(safe_float(cr.get("pubs_pct_of_parent")) * 100, 1),
                "FWCI median": round(safe_float(cr.get("fwci_median")), 2),
                "% Int'l": round(safe_float(cr.get("pct_international")) * 100, 1),
                "CAGR": format_cagr(cr.get(CAGR_COL)),
            })
        df_tbl_c = pd.DataFrame(tbl_children).sort_values("Pubs", ascending=False)
        st.dataframe(
            df_tbl_c,
            use_container_width=True,
            hide_index=True,
            height=min(500, len(df_tbl_c) * 38 + 50),
            column_config={
                "% of Parent": st.column_config.ProgressColumn(
                    "% of Parent", min_value=0, max_value=100, format="%.1f%%",
                ),
                "% Int'l": st.column_config.ProgressColumn(
                    "% Int'l", min_value=0, max_value=100, format="%.1f%%",
                ),
            },
        )
    else:
        st.info(f"No {child_label.lower()} data available for this element.")

    st.divider()


# =====================================================================
#  5. Top Partners
# =====================================================================

st.header("Collaboration Partners")

partner_row = get_partners_for_element(element_level, element_id)

if partner_row is not None:

    # ── Find the right blob columns ──────────────────────────────
    # Use exact column names from the actual data, with fallback search
    int_blob_col = None
    dom_blob_col = None
    recip_blob_col = None

    for col in partner_row.index:
        col_lower = col.lower().replace(" ", "_")
        if col == "top_int_partners" or "top_int_partner" in col_lower:
            int_blob_col = col
        # Domestic: look for the configured keyword (default: "top_domestic_partners")
        if col == DOMESTIC_PARTNER_COL_KEYWORD or "top_domestic" in col_lower:
            dom_blob_col = col
        if col == "reciprocity_partners" or "reciprocity" in col_lower:
            recip_blob_col = col

    # ── Tabs ──────────────────────────────────────────────────────
    tab_int, tab_dom, tab_recip = st.tabs([
        "Top International Partners",
        f"Top {DOMESTIC_LABEL} Partners",
        "Reciprocity Analysis",
    ])

    with tab_int:
        if int_blob_col:
            partners_int = parse_int_partners(partner_row.get(int_blob_col, ""))
            if partners_int:
                df_int = pd.DataFrame(partners_int)
                df_int["copubs"] = df_int["copubs"].apply(safe_int)
                df_int["fwci"] = df_int["fwci"].apply(safe_float)
                df_int["share_inst_fmt"] = df_int["share_inst"].apply(
                    lambda x: f"{safe_float(x) * 100:.1f}%"
                )
                df_int["share_partner_fmt"] = df_int["share_partner"].apply(
                    lambda x: f"{safe_float(x) * 100:.1f}%"
                )
                df_int["fwci_fmt"] = df_int["fwci"].apply(lambda x: f"{x:.2f}")
                display_cols = {
                    "name": "Partner",
                    "country": "Country",
                    "type": "Type",
                    "copubs": "Co-pubs",
                    "share_inst_fmt": f"% of {SHORT_NAME}",
                    "share_partner_fmt": "% of Partner",
                    "fwci_fmt": "Avg FWCI",
                }
                df_show = df_int.rename(columns=display_cols)[list(display_cols.values())]
                st.dataframe(df_show, use_container_width=True, hide_index=True)
            else:
                st.info("No international partner data available.")
        else:
            st.info("No international partner data available.")

    # ── Domestic partners ─────────────────────────────────────────
    with tab_dom:
        if dom_blob_col:
            # Domestic uses SAME format as international (includes country)
            partners_dom = parse_domestic_partners(partner_row.get(dom_blob_col, ""))
            if partners_dom:
                df_dom = pd.DataFrame(partners_dom)
                df_dom["copubs"] = df_dom["copubs"].apply(safe_int)
                df_dom["fwci"] = df_dom["fwci"].apply(safe_float)
                df_dom["share_inst_fmt"] = df_dom["share_inst"].apply(
                    lambda x: f"{safe_float(x) * 100:.1f}%"
                )
                df_dom["fwci_fmt"] = df_dom["fwci"].apply(lambda x: f"{x:.2f}")
                display_cols_d = {
                    "name": "Partner",
                    "type": "Type",
                    "copubs": "Co-pubs",
                    "share_inst_fmt": f"% of {SHORT_NAME}",
                    "fwci_fmt": "Avg FWCI",
                }
                df_show_d = df_dom.rename(columns=display_cols_d)[list(display_cols_d.values())]
                st.dataframe(df_show_d, use_container_width=True, hide_index=True)
            else:
                st.info(f"No {DOMESTIC_LABEL} partner data available.")
        else:
            st.info(f"No {DOMESTIC_LABEL} partner data available.")

    # ── Reciprocity scatter ───────────────────────────────────────
    with tab_recip:
        if recip_blob_col:
            partners_recip = parse_reciprocity_partners(partner_row.get(recip_blob_col, ""))
            if partners_recip:
                df_recip = pd.DataFrame(partners_recip)
                df_recip["copubs"] = df_recip["copubs"].apply(safe_int)
                df_recip["share_inst"] = df_recip["share_inst"].apply(safe_float)
                df_recip["share_partner"] = df_recip["share_partner"].apply(safe_float)
                df_recip["partner_total"] = df_recip["partner_total"].apply(safe_int)
                df_recip["fwci"] = df_recip["fwci"].apply(safe_float)

                # Geo category
                def geo_cat(country: str) -> str:
                    return DOMESTIC_COUNTRY if str(country).strip() == DOMESTIC_COUNTRY else "International"

                df_recip["geo"] = df_recip["country"].apply(geo_cat)

                # Filter controls
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    remove_outliers = st.checkbox(
                        "Remove outliers (share > 100%)", value=True, key="recip_outlier"
                    )
                with col_r2:
                    n_partners = st.slider(
                        "Number of partners",
                        min_value=5,
                        max_value=min(100, len(df_recip)),
                        value=min(30, len(df_recip)),
                        key="recip_n",
                    )

                df_plot = df_recip.nlargest(n_partners, "copubs")
                if remove_outliers:
                    df_plot = df_plot[
                        (df_plot["share_inst"] <= 1.0) & (df_plot["share_partner"] <= 1.0)
                    ]

                if not df_plot.empty:
                    fig_recip = plot_reciprocity_scatter(
                        df_plot,
                        domestic_country=DOMESTIC_COUNTRY,
                        short_name=SHORT_NAME,
                        element_name=element_name,
                    )
                    st.plotly_chart(fig_recip, use_container_width=True)
                    st.caption(
                        "Bubble size = partner's total publications in this element. "
                        "Dashed line = balanced dependency."
                    )
                else:
                    st.info("No data to display after filtering.")
            else:
                st.info("No reciprocity data available.")
        else:
            st.info("No reciprocity data available.")

else:
    st.info("No partner data available for this element.")

st.divider()

# =====================================================================
#  6. Top Authors
# =====================================================================

st.header("Top Authors")

author_row = get_authors_for_element(element_level, element_id)

if author_row is not None:
    # Find the author blob column
    author_blob_col = None
    for col in author_row.index:
        if col == "top_authors" or "top_authors" in col.lower().replace(" ", "_"):
            author_blob_col = col
            break

    if author_blob_col:
        authors = parse_authors(author_row.get(author_blob_col, ""))
        if authors:
            df_auth = pd.DataFrame(authors)
            df_auth["pubs_int"] = df_auth["pubs"].apply(safe_int)
            df_auth["fwci_float"] = df_auth["fwci"].apply(safe_float)
            df_auth["pct_fmt"] = df_auth["pct"].apply(
                lambda x: f"{safe_float(x) * 100:.1f}%"
            )
            df_auth["fwci_fmt"] = df_auth["fwci_float"].apply(lambda x: f"{x:.2f}")

            display_cols_a = {
                "name": "Author",
                "pubs_int": "Publications",
                "pct_fmt": f"% of {element_name}",
                "fwci_fmt": "Avg FWCI",
            }

            df_show_a = df_auth.rename(columns=display_cols_a)[list(display_cols_a.values())]
            st.dataframe(df_show_a, use_container_width=True, hide_index=True)
        else:
            st.info("No author data available.")
    else:
        st.info("No author data available.")
else:
    st.info("No author data available for this element.")
