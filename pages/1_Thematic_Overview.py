# pages/3___Thematic_Overview.py
# Thematic Overview — UAM research portfolio at a glance

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Thematic Overview — UAM", page_icon="🏛️", layout="wide")

from lib.config import (
    SHORT_NAME,
    CAGR_COL,
    CAGR_LABEL,
    DOMAIN_COLORS,
    DOMAIN_NAMES_ORDERED,
    DOMAIN_ORDER,
    OA_TOPIC_LEVEL_NAME,
)
from lib.data_cache import (
    get_overview_for_level,
    load_treemap_hierarchy,
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
    parse_fwci_boxplot,
    plot_horizontal_bar,
    plot_precomputed_boxplot,
    plot_treemap,
    pubs_pct_col,
    render_domain_legend,
    safe_float,
    safe_int,
)

# ── Header ────────────────────────────────────────────────────────────

st.title("🔬 Thematic Overview")
st.caption(f"Research portfolio of **{SHORT_NAME}** across the OpenAlex taxonomy.")
st.markdown(render_domain_legend(), unsafe_allow_html=True)
st.divider()

# =====================================================================
#  1. Treemap
# =====================================================================

st.header("Research Portfolio Treemap")

df_tree = load_treemap_hierarchy()

if df_tree is not None and not df_tree.empty:
    # Colour metric selector — only offer metrics that exist in the data
    available_metrics = {}
    if "fwci_median" in df_tree.columns:
        available_metrics["FWCI (median)"] = "fwci_median"
    if "pct_international" in df_tree.columns:
        available_metrics["% International"] = "pct_international"
    if "cagr" in df_tree.columns:
        available_metrics["CAGR"] = "cagr"

    if available_metrics:
        col_tm1, col_tm2 = st.columns([3, 1])
        with col_tm2:
            color_choice = st.selectbox("Colour by", list(available_metrics.keys()), index=0)
        color_metric = available_metrics[color_choice]
    else:
        color_metric = "fwci_median"

    fig_tree = plot_treemap(df_tree, color_metric=color_metric)
    st.plotly_chart(fig_tree, use_container_width=True)
else:
    st.info("No treemap data available.")

st.divider()

# =====================================================================
#  2. Domain-Level Overview
# =====================================================================

st.header("Domains")

df_domains = get_overview_for_level("domain")

if not df_domains.empty:
    # ── KPI row ───────────────────────────────────────────────────
    total_pubs = df_domains["pubs_total"].sum()
    st.metric("Total indexed publications", format_int(total_pubs))
    st.write("")

    # ── Table ─────────────────────────────────────────────────────
    tbl = build_overview_table(df_domains, "domain", show_emoji=True)
    tbl = tbl.sort_values("Pubs", ascending=False)

    st.dataframe(
        tbl,
        use_container_width=True,
        hide_index=True,
        height=min(400, len(tbl) * 45 + 50),
        column_config={
            "% Total": st.column_config.ProgressColumn(
                "% Total", min_value=0, max_value=100, format="%.1f%%",
            ),
        },
    )

    # ── FWCI Boxplot by domain ────────────────────────────────────
    st.subheader("FWCI Distribution by Domain")
    show_extremes = st.toggle("Include extreme values (p0, p100)", value=False, key="dom_ext")

    bp_data = []
    for _, r in df_domains.iterrows():
        bp = parse_fwci_boxplot(r.get("fwci_boxplot", ""))
        if bp is None:
            continue
        dname = r.get("name", "")
        bp["category"] = dname
        bp["color"] = domain_color(dname)
        bp["count"] = safe_int(r.get("pubs_total"))
        bp_data.append(bp)

    if bp_data:
        fig_bp = plot_precomputed_boxplot(bp_data, show_extremes=show_extremes, height=500)
        fig_bp.update_layout(yaxis_title="FWCI")
        st.plotly_chart(fig_bp, use_container_width=True)
    else:
        st.info("No FWCI boxplot data available at domain level.")

else:
    st.info("No domain-level data available.")

st.divider()

# =====================================================================
#  3. Field-Level Overview
# =====================================================================

st.header("Fields (OpenAlex L2)")

df_fields = get_overview_for_level("field")

if not df_fields.empty:
    # ── Filter / search ───────────────────────────────────────────
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        domain_filter = st.multiselect(
            "Filter by domain",
            options=DOMAIN_NAMES_ORDERED,
            default=[],
            key="field_dom_filter",
        )
    with col_f2:
        search_field = st.text_input("Search field name", "", key="field_search")

    df_fields_show = df_fields.copy()
    if domain_filter:
        domain_ids = [
            DOMAIN_ORDER[DOMAIN_NAMES_ORDERED.index(d)]
            for d in domain_filter
            if d in DOMAIN_NAMES_ORDERED
        ]
        df_fields_show = df_fields_show[df_fields_show["domain_id"].isin(domain_ids)]
    if search_field:
        df_fields_show = df_fields_show[
            df_fields_show["name"].str.contains(search_field, case=False, na=False)
        ]

    tbl_f = build_overview_table(df_fields_show, "field", show_emoji=True)
    tbl_f = tbl_f.sort_values("Pubs", ascending=False)

    st.dataframe(
        tbl_f,
        use_container_width=True,
        hide_index=True,
        height=min(800, len(tbl_f) * 38 + 50),
        column_config={
            "% Total": st.column_config.ProgressColumn(
                "% Total", min_value=0, max_value=100, format="%.1f%%",
            ),
        },
    )

    # ── FWCI Boxplot by field ─────────────────────────────────────
    st.subheader("FWCI Distribution by Field")
    show_extremes_f = st.toggle("Include extreme values (p0, p100)", value=False, key="field_ext")

    bp_field_data = []
    for _, r in df_fields_show.iterrows():
        bp = parse_fwci_boxplot(r.get("fwci_boxplot", ""))
        if bp is None:
            continue
        dname = domain_name_for_id(r.get("domain_id", 0))
        bp["category"] = r.get("name", "")
        bp["color"] = domain_color(dname)
        bp["count"] = safe_int(r.get("pubs_total"))
        bp_field_data.append(bp)

    if bp_field_data:
        fig_bp_f = plot_precomputed_boxplot(
            bp_field_data,
            show_extremes=show_extremes_f,
            height=max(500, len(bp_field_data) * 25 + 80),
        )
        fig_bp_f.update_layout(
            xaxis_tickangle=-45,
            yaxis_title="FWCI",
            margin=dict(b=160),
        )
        st.plotly_chart(fig_bp_f, use_container_width=True)
    else:
        st.info("No FWCI boxplot data available at field level.")

else:
    st.info("No field-level data available.")

st.divider()

# =====================================================================
#  4. Subfield-Level Overview (compact table)
# =====================================================================

st.header("Subfields (OpenAlex L3)")

df_subfields = get_overview_for_level("subfield")

if not df_subfields.empty:
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        sf_domain_filter = st.multiselect(
            "Filter by domain",
            options=DOMAIN_NAMES_ORDERED,
            default=[],
            key="sf_dom_filter",
        )
    with col_s2:
        sf_search = st.text_input("Search subfield name", "", key="sf_search")

    df_sf_show = df_subfields.copy()
    if sf_domain_filter:
        domain_ids = [
            DOMAIN_ORDER[DOMAIN_NAMES_ORDERED.index(d)]
            for d in sf_domain_filter
            if d in DOMAIN_NAMES_ORDERED
        ]
        df_sf_show = df_sf_show[df_sf_show["domain_id"].isin(domain_ids)]
    if sf_search:
        df_sf_show = df_sf_show[
            df_sf_show["name"].str.contains(sf_search, case=False, na=False)
        ]

    tbl_sf = build_overview_table(df_sf_show, "subfield", show_emoji=True)
    tbl_sf = tbl_sf.sort_values("Pubs", ascending=False)

    st.dataframe(
        tbl_sf,
        use_container_width=True,
        hide_index=True,
        height=600,
        column_config={
            "% Total": st.column_config.ProgressColumn(
                "% Total", min_value=0, max_value=100, format="%.1f%%",
            ),
        },
    )
else:
    st.info("No subfield-level data available.")

st.divider()

# =====================================================================
#  5. Topic-Level Overview (OpenAlex L4, compact)
# =====================================================================

st.header("Topics (OpenAlex L4)")

df_topics = get_overview_for_level(OA_TOPIC_LEVEL_NAME)

if not df_topics.empty:
    col_t1, col_t2, col_t3 = st.columns([1, 1, 1])
    with col_t1:
        tp_domain_filter = st.multiselect(
            "Filter by domain",
            options=DOMAIN_NAMES_ORDERED,
            default=[],
            key="tp_dom_filter",
        )
    with col_t2:
        tp_search = st.text_input("Search topic name", "", key="tp_search")
    with col_t3:
        tp_top_n = st.slider("Show top N topics", 10, 200, 50, key="tp_topn")

    df_tp_show = df_topics.copy()
    if tp_domain_filter:
        domain_ids = [
            DOMAIN_ORDER[DOMAIN_NAMES_ORDERED.index(d)]
            for d in tp_domain_filter
            if d in DOMAIN_NAMES_ORDERED
        ]
        df_tp_show = df_tp_show[df_tp_show["domain_id"].isin(domain_ids)]
    if tp_search:
        df_tp_show = df_tp_show[
            df_tp_show["name"].str.contains(tp_search, case=False, na=False)
        ]

    df_tp_show = df_tp_show.nlargest(tp_top_n, "pubs_total")
    tbl_tp = build_overview_table(df_tp_show, OA_TOPIC_LEVEL_NAME, show_emoji=True)
    if not tbl_tp.empty:
        tbl_tp = tbl_tp.sort_values("Pubs", ascending=False)
        st.dataframe(
            tbl_tp,
            use_container_width=True,
            hide_index=True,
            height=600,
            column_config={
                "% Total": st.column_config.ProgressColumn(
                    "% Total", min_value=0, max_value=100, format="%.1f%%",
                ),
            },
        )
    else:
        st.info("No topics match the current filters.")

    st.dataframe(
        tbl_tp,
        use_container_width=True,
        hide_index=True,
        height=600,
        column_config={
            "% Total": st.column_config.ProgressColumn(
                "% Total", min_value=0, max_value=100, format="%.1f%%",
            ),
        },
    )
else:
    st.info("No topic-level data available.")
