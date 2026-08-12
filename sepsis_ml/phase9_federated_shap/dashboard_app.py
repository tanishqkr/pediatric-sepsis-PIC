"""
dashboard_app.py

Minimal Streamlit dashboard for the mentor's requirement: top-5 features +
visualization, node-level comparison, sensitivity results. Reads ONLY
dashboard.json (produced by export_dashboard_data.py) — nothing else needs
to travel to the demo machine.

Run with:
    streamlit run dashboard_app.py -- --data path/to/dashboard.json
"""
import argparse
import json
import sys

import pandas as pd
import streamlit as st

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="dashboard.json")
# streamlit passes its own args first; grab only what's after "--"
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])

st.set_page_config(page_title="Federated SHAP — Pediatric Sepsis", layout="wide")

with open(args.data) as f:
    data = json.load(f)

st.title("Federated SHAP & Sensitivity Dashboard")
st.caption(f"Run: {data['run_id']}")

algo = st.sidebar.radio("Federated algorithm", ["fedavg", "fedprox"])
section = data.get(algo)

st.sidebar.markdown("### Nodes")
for node, n in data["nodes"].items():
    st.sidebar.write(f"{node}: {n} patients")

if section is None:
    st.warning(f"No results yet for {algo} — run the pipeline first.")
else:
    st.header(f"Top {data['top_k']} Features — {algo.upper()}")
    top5_df = pd.DataFrame(section["top_5_features"])
    top5_df["display_name"] = top5_df.get("display_name", top5_df["feature"])
    c1, c2 = st.columns([2, 1])
    with c1:
        st.bar_chart(top5_df.set_index("feature")["federated_mean_abs_shap"])
    with c2:
        st.dataframe(top5_df, use_container_width=True)

    st.header("Node Agreement With the Federated Ranking")
    st.caption("Low correlation = that node's local explanation diverges from the aggregate.")
    corr_df = pd.DataFrame(section["node_vs_federated_spearman"]).T.reset_index()
    corr_df.columns = ["node", "spearman_rho", "p_value"]
    st.dataframe(corr_df, use_container_width=True)
    st.bar_chart(corr_df.set_index("node")["spearman_rho"])

    with st.expander("Full federated feature ranking"):
        st.dataframe(pd.DataFrame(section["federated_ranking"]), use_container_width=True)

st.header("Sensitivity Analysis")
sens = data.get("sensitivity_analysis")
if not sens:
    st.info("Sensitivity analysis not yet run.")
else:
    for axis_name, axis_result in sens.items():
        st.subheader(axis_name)
        st.json(axis_result)
