# lib/config.py — Uniwersytet im. Adama Mickiewicza w Poznaniu (UAM)

# ── Institution Identity ──────────────────────────────────────────────
INSTITUTION_NAME = "Uniwersytet im. Adama Mickiewicza w Poznaniu"
SHORT_NAME = "UAM"
SHORT_CODE = "uam"
OPENALEX_ID = "I162498740"
PAGE_ICON = "🏛️"

# ── Time Range ────────────────────────────────────────────────────────
YEAR_START = 2020
YEAR_END = 2024
YEARS = list(range(YEAR_START, YEAR_END + 1))
CAGR_COL = "cagr_2020_2024"
CAGR_LABEL = "2020-24"

# ── Geography ─────────────────────────────────────────────────────────
DOMESTIC_COUNTRY = "Poland"
DOMESTIC_LABEL = "Polish"
DOMESTIC_CODE = "pl"

# ── Initiative / Excellence Programme ─────────────────────────────────
INITIATIVE_LABEL = None
INITIATIVE_COL = None
INITIATIVE_COUNT_COL = None

# ── National / Advanced Metrics ───────────────────────────────────────
SI_NATIONAL_COL = None
SI_NATIONAL_LABEL = None
HAS_SI = False
HAS_NCI = False
HAS_DOMINANCE = False
HAS_PP_TOP = False
HAS_COPUBS_LINKS = False

# ── Internal Structures ──────────────────────────────────────────────
HAS_LAB_VIEW = False
STRUCTURE_TYPES = []
STRUCTURE_TYPE_COLORS = {}
HAS_POLES = False

# ── Topic Modeling ────────────────────────────────────────────────────
# No custom topic model for UAM — set to None to skip TM sections
TM_LEVEL_NAME = None
OA_TOPIC_LEVEL_NAME = "topic"
TM_METHOD_DESCRIPTION = None
TM_LABELS_DIMENSION = None

# ── SDG ───────────────────────────────────────────────────────────────
HAS_SDG = False

# ── Column name overrides ────────────────────────────────────────────
# SCHEMAS.md says pubs_pct_of_{code} but actual data uses pubs_pct_of_total.
# This constant is the ACTUAL column name in the parquet files.
PUBS_PCT_COL = "pubs_pct_of_total"

# Domestic partner column is "top_domestic_partners" (not "top_pl_partners")
# and uses the SAME 9-field blob format as international partners.
DOMESTIC_PARTNER_COL_KEYWORD = "top_domestic_partners"

# ── Domain Configuration (constant across all apps) ──────────────────
DOMAIN_ORDER = [1, 2, 3, 4]
DOMAIN_NAMES_ORDERED = [
    "Life Sciences",
    "Social Sciences",
    "Physical Sciences",
    "Health Sciences",
]

DOMAIN_COLORS = {
    1: "#0CA750",
    2: "#FFCB3A",
    3: "#8190FF",
    4: "#F85C32",
    "Life Sciences": "#0CA750",
    "Social Sciences": "#FFCB3A",
    "Physical Sciences": "#8190FF",
    "Health Sciences": "#F85C32",
    "Other": "#7f7f7f",
}

DOMAIN_EMOJI = {
    "Health Sciences": "🟥",
    "Life Sciences": "🟩",
    "Physical Sciences": "🟦",
    "Social Sciences": "🟨",
    "Other": "⬜",
}

# ── Document Type Configuration (constant) ───────────────────────────
DOCTYPE_COLORS = {
    "Articles": "#4285F4",
    "Book chapters": "#FBBC05",
    "Books": "#EA4335",
    "Reviews": "#34A853",
    "Preprints": "#9E9E9E",
}

# ── OpenAlex Document Types Filter ───────────────────────────────────
OA_DOC_TYPES = "types/article|types/book-chapter|types/book|types/review"
OA_DOC_TYPES_EXTRA = []
