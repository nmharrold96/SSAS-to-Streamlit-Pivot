# app.py
# Streamlit SSAS Cube Explorer — dynamic MDX builder
# -------------------------------------------------
# Requirements (install in your environment):
#   pip install streamlit pyadomd pandas streamlit-aggrid pivottablejs python-dotenv
# Run:  streamlit run app.py

import os
import io
import pandas as pd
import streamlit as st
from typing import List, Tuple

# Optional: load connection string from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Lazy import these so the app still loads if packages are missing
try:
    from pyadomd import Pyadomd
except Exception as e:
    Pyadomd = None
    _pyadomd_err = e

try:
    from st_aggrid import AgGrid, GridOptionsBuilder
except Exception:
    AgGrid = None
    GridOptionsBuilder = None

try:
    from pivottablejs import pivot_ui
except Exception:
    pivot_ui = None

st.set_page_config(page_title="SSAS Cube Explorer", layout="wide")
st.title("🧊 SSAS Cube Explorer — Dynamic MDX in Streamlit")
st.caption("Connect to an SSAS cube, pick measures & levels, auto-generate MDX, and explore interactively.")

# ----------------------------
# Sidebar: Connection settings
# ----------------------------
with st.sidebar:
    st.header("Connection")
    default_conn = os.getenv("SSAS_CONNECTION_STRING", "Provider=MSOLAP;Data Source=YOUR_SERVER;Initial Catalog=YOUR_DB;")
    connection_str = st.text_area("SSAS connection string", value=default_conn, height=80, help=(
        "Typical: Provider=MSOLAP;Data Source=ServerName;Initial Catalog=Database;"
        " Optionally include user/password or integrated security."))

    st.markdown(
        "**Tip:** You can also connect to Tabular over XMLA endpoints. If using Azure AS/Power BI, "
        "use the proper connection string for your workspace/server.")

    connect = st.button("🔌 Connect / Refresh Metadata", use_container_width=True)

if Pyadomd is None:
    st.error(f"pyadomd not available: {_pyadomd_err}\nRun: pip install pyadomd")
    st.stop()

# ----------------------------
# Helpers & cache
# ----------------------------
@st.cache_resource(show_spinner=False)
def _open_connection(conn_str: str):
    return Pyadomd(conn_str)

@st.cache_data(show_spinner=True)
def get_schema_rowsets(conn_str: str):
    """Return cubes, measures, dimensions, hierarchies, levels as DataFrames."""
    with _open_connection(conn_str) as conn:
        cubes = pd.DataFrame(conn.get_schema_rowset("MDSCHEMA_CUBES"))
        measures = pd.DataFrame(conn.get_schema_rowset("MDSCHEMA_MEASURES"))
        dims = pd.DataFrame(conn.get_schema_rowset("MDSCHEMA_DIMENSIONS"))
        hier = pd.DataFrame(conn.get_schema_rowset("MDSCHEMA_HIERARCHIES"))
        lvls = pd.DataFrame(conn.get_schema_rowset("MDSCHEMA_LEVELS"))
    # Ensure expected columns exist (some providers differ slightly)
    for df, needed in [
        (cubes, ["CUBE_NAME"]),
        (measures, ["MEASURE_NAME", "MEASURE_UNIQUE_NAME", "MEASUREGROUP_NAME", "CUBE_NAME"]),
        (dims, ["DIMENSION_NAME", "DIMENSION_UNIQUE_NAME", "CUBE_NAME", "DIMENSION_TYPE"]),
        (hier, ["HIERARCHY_NAME", "HIERARCHY_UNIQUE_NAME", "DIMENSION_UNIQUE_NAME", "CUBE_NAME", "HIERARCHY_CAPTION"]),
        (lvls, ["LEVEL_NAME", "LEVEL_UNIQUE_NAME", "HIERARCHY_UNIQUE_NAME", "LEVEL_CAPTION", "LEVEL_NUMBER", "CUBE_NAME"])]:
        for col in needed:
            if col not in df.columns:
                df[col] = None
    return cubes, measures, dims, hier, lvls

# ----------------------------
# UI: Metadata selection
# ----------------------------
if connect:
    st.session_state["meta"] = None

if "meta" not in st.session_state:
    st.session_state["meta"] = None

if st.session_state["meta"] is None and connection_str:
    try:
        with st.spinner("Connecting and loading metadata…"):
            st.session_state["meta"] = get_schema_rowsets(connection_str)
        st.success("Metadata loaded.")
    except Exception as e:
        st.exception(e)
        st.stop()

if st.session_state["meta"] is None:
    st.info("Enter a valid SSAS connection string in the left sidebar, then click **Connect / Refresh Metadata**.")
    st.stop()

cubes, measures, dims, hier, lvls = st.session_state["meta"]

# Filter metadata by selected cube
cube_names = sorted(cubes["CUBE_NAME"].dropna().unique().tolist())
selected_cube = st.selectbox("Cube", options=cube_names, index=0 if cube_names else None)

measures_cube = measures[measures["CUBE_NAME"] == selected_cube].copy()
levels_cube = lvls[lvls["CUBE_NAME"] == selected_cube].copy()
hier_cube = hier[hier["CUBE_NAME"] == selected_cube].copy()

# Friendly labels for UI
measures_cube["_label"] = measures_cube["MEASURE_NAME"].fillna("") + "  ·  " + measures_cube["MEASUREGROUP_NAME"].fillna("")
levels_cube["_label"] = hier_cube.set_index("HIERARCHY_UNIQUE_NAME").loc[levels_cube["HIERARCHY_UNIQUE_NAME"].values]["HIERARCHY_CAPTION"].values
levels_cube["_label"] = levels_cube["_label"].fillna("") + " › " + levels_cube["LEVEL_CAPTION"].fillna(levels_cube["LEVEL_NAME"]).astype(str)

cols = st.columns(3)
with cols[0]:
    measures_mult = st.multiselect(
        "Measures (put on Columns)",
        options=measures_cube["MEASURE_UNIQUE_NAME"].tolist(),
        format_func=lambda u: measures_cube.loc[measures_cube["MEASURE_UNIQUE_NAME"]==u, "_label"].iloc[0]
        if (measures_cube["MEASURE_UNIQUE_NAME"]==u).any() else str(u),
    )
with cols[1]:
    rows_levels = st.multiselect(
        "Rows: pick Levels (supports multi, will CrossJoin)",
        options=levels_cube["LEVEL_UNIQUE_NAME"].tolist(),
        format_func=lambda u: levels_cube.loc[levels_cube["LEVEL_UNIQUE_NAME"]==u, "_label"].iloc[0]
        if (levels_cube["LEVEL_UNIQUE_NAME"]==u).any() else str(u),
    )
with cols[2]:
    cols_levels = st.multiselect(
        "Columns: pick Levels (optional)",
        options=levels_cube["LEVEL_UNIQUE_NAME"].tolist(),
        format_func=lambda u: levels_cube.loc[levels_cube["LEVEL_UNIQUE_NAME"]==u, "_label"].iloc[0]
        if (levels_cube["LEVEL_UNIQUE_NAME"]==u).any() else str(u),
    )

adv1, adv2, adv3 = st.columns([1,1,1])
with adv1:
    top_n_rows = st.number_input("Limit Rows (TOPCOUNT)", min_value=0, max_value=100000, value=500, step=50,
                                 help="0 = no limit. Applied after NON EMPTY.")
with adv2:
    where_expr = st.text_input("WHERE (optional MDX tuple or slicer set)", value="",
                               placeholder="e.g. ([Date].[Calendar].[Calendar Year].&[2024])")
with adv3:
    non_empty = st.checkbox("Use NON EMPTY on axes", value=True)

# --------------------------------------------------
# MDX builder
# --------------------------------------------------
def _level_members(level_unique_name: str) -> str:
    """Return MDX set expression for a level's members."""
    # Example: [Date].[Calendar].[Calendar Year].Members
    return f"{level_unique_name}.Members"

def _crossjoin(sets: List[str]) -> str:
    if not sets:
        return "{}"
    if len(sets) == 1:
        return sets[0]
    return "CrossJoin(" + ", ".join(sets) + ")"

def _apply_non_empty(set_expr: str) -> str:
    return f"NON EMPTY {set_expr}" if non_empty else set_expr

def build_axis_set(level_unique_names: List[str]) -> str:
    sets = [_level_members(u) for u in level_unique_names]
    return _apply_non_empty(_crossjoin(sets)) if sets else "{}"

def build_measures_set(measure_unique_names: List[str]) -> str:
    if not measure_unique_names:
        return "{}"
    # Ensure each is wrapped in braces if needed — they should already be unique names like [Measures].[X]
    measures_set = "{ " + ", ".join(measure_unique_names) + " }"
    return _apply_non_empty(measures_set)

def build_mdx(cube_name: str,
              measures_u: List[str],
              rows_levels_u: List[str],
              cols_levels_u: List[str],
              where_expr: str = "",
              top_n_rows: int = 0) -> str:
    rows_set = build_axis_set(rows_levels_u)

    # Optionally TOPCOUNT the rows (after non-empty)
    if top_n_rows and rows_set and rows_set != "{}":
        rows_set = f"TopCount({rows_set}, {top_n_rows})"

    # Columns: combine measures and any column levels via CrossJoin
    measures_set = build_measures_set(measures_u)
    cols_set_levels = build_axis_set(cols_levels_u)

    if cols_set_levels and cols_set_levels != "{}":
        if measures_set and measures_set != "{}":
            columns_set = _apply_non_empty(_crossjoin([measures_set, cols_set_levels]))
        else:
            columns_set = cols_set_levels
    else:
        columns_set = measures_set if measures_set else "{}"

    where_clause = f" WHERE ({where_expr})" if where_expr.strip() else ""

    mdx = f"""
SELECT
  {columns_set} ON COLUMNS,
  {rows_set} ON ROWS
FROM [{cube_name}]
{where_clause}
""".strip()
    return mdx

# Build MDX preview
mdx_preview = build_mdx(
    cube_name=selected_cube,
    measures_u=measures_mult,
    rows_levels_u=rows_levels,
    cols_levels_u=cols_levels,
    where_expr=where_expr,
    top_n_rows=int(top_n_rows),
)

with st.expander("🧱 Generated MDX", expanded=True):
    st.code(mdx_preview, language="mdx")

# --------------------------------------------------
# Execute MDX and display
# --------------------------------------------------
@st.cache_data(show_spinner=True)
def run_mdx(conn_str: str, mdx: str) -> pd.DataFrame:
    with _open_connection(conn_str) as conn:
        with conn.cursor().execute(mdx) as cur:
            cols = [c.name for c in cur.description]
            data = cur.fetchall()
    df = pd.DataFrame(data, columns=cols)
    return df

run_query = st.button("▶️ Run Query", type="primary", use_container_width=True)

if run_query:
    if not measures_mult and not cols_levels:
        st.warning("Select at least one Measure (or put a Level on Columns).")
    if not rows_levels:
        st.warning("Select at least one Level on Rows to shape the result set.")

    try:
        with st.spinner("Executing MDX…"):
            df = run_mdx(connection_str, mdx_preview)
        if df.empty:
            st.info("No rows returned.")
        else:
            st.success(f"Returned {len(df):,} rows · {len(df.columns)} columns")

            # Download CSV
            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=False)
            st.download_button("⬇️ Download CSV", data=csv_buf.getvalue(), file_name="ssas_result.csv", mime="text/csv")

            # Tabs for different viewers
            t1, t2 = st.tabs(["AG Grid (group/pivot)", "PivotTable.js (drag & drop)"])

            with t1:
                if AgGrid is None or GridOptionsBuilder is None:
                    st.error("st-aggrid not installed. Run: pip install streamlit-aggrid")
                else:
                    gb = GridOptionsBuilder.from_dataframe(df)
                    gb.configure_pagination(paginationAutoPageSize=True)
                    gb.configure_side_bar()
                    # Enable grouping / pivot / aggregation like a spreadsheet
                    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, enablePivot=True, enableValue=True, aggFunc='sum', editable=False)
                    grid_options = gb.build()
                    AgGrid(df, gridOptions=grid_options, enableEnterpriseModules=True, height=600)

            with t2:
                if pivot_ui is None:
                    st.error("pivottablejs not installed. Run: pip install pivottablejs")
                else:
                    # Generate pivot HTML to a temp file and embed
                    tmp_path = os.path.join(st.experimental_get_query_params().get('tmpdir', ['.'])[0], "_pivot.html")
                    try:
                        pivot_ui(df, outfile_path=tmp_path)
                        with open(tmp_path, "r", encoding="utf-8") as f:
                            html = f.read()
                        st.components.v1.html(html, height=650, scrolling=True)
                    except Exception as e:
                        st.exception(e)

    except Exception as e:
        st.exception(e)

# --------------------------------------------------
# Notes & Tips
# --------------------------------------------------
with st.expander("ℹ️ Tips & Notes"):
    st.markdown(
        """
- **Metadata scope**: This app lists **Levels**; dragging multiple levels to *Rows* or *Columns* will `CrossJoin` their member sets.
- **Measures on Columns**: Measures are placed on the Columns axis. If you also pick column levels, we `CrossJoin` measures with those levels.
- **Filters**: Use the `WHERE` box for slicers, e.g. `([Date].[Calendar].[Calendar Year].&[2024])` or a set.
- **Performance**: Start with a small `TOPCOUNT` (e.g. 500–2000). Remove it after you’re sure the shape is right.
- **NON EMPTY**: Enabled by default to remove empty tuples on axes.
- **Azure / XMLA**: For Azure AS / Power BI XMLA, ensure your connection string and credentials are correct for XMLA read.
- **Security**: Avoid storing secrets in code. Use environment variables or Streamlit secrets.
        """
    )

with st.sidebar:
    st.markdown("---")
    st.caption("Built for exploration. Validate MDX and security before production use.")
