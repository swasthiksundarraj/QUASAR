from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import pandas as pd
import streamlit as st

st.set_page_config(page_title="QUASAR Research Dashboard", layout="wide")
st.title("QATNet + QUASAR")
st.warning("Synthetic telemetry and simulated mitigation only; no physical QKD or live network actions.")

root = Path(__file__).resolve().parents[2]
demo_file = root / "outputs" / "demo" / "demo.json"
metrics_file = root / "outputs" / "metrics" / "results.csv"

if st.button("Run deterministic mock demonstration"):
    from quasar.demo import run_demo
    run_demo(root / "outputs" / "demo")
    st.rerun()

if demo_file.exists():
    demo = json.loads(demo_file.read_text())
    assessment = demo["assessment"]
    left, middle, right = st.columns(3)
    left.metric("Cross-layer risk", f'{assessment["risk_score"]:.2f}')
    middle.metric("Risk level", assessment["risk_level"])
    right.metric("Verifier", "Approved" if demo["verifier"]["accepted"] else "Fallback")
    st.subheader("QKD and key-pool telemetry")
    telemetry_path = root / "outputs" / "qkd" / "telemetry.csv"
    if not telemetry_path.exists():
        telemetry_path = root / "outputs" / "demo" / "qkd-telemetry.csv"
    if telemetry_path.exists():
        qkd = pd.read_csv(telemetry_path)
        indexed = qkd.set_index("timestamp")
        qber_chart, pool_chart, rate_chart = st.columns(3)
        qber_chart.line_chart(indexed[["qber"]], height=220)
        qber_chart.caption("QBER")
        pool_chart.line_chart(indexed[["key_pool_bits"]], height=220)
        pool_chart.caption("Key pool (bits)")
        rate_chart.line_chart(indexed[["secret_key_rate_bps"]], height=220)
        rate_chart.caption("Secret key rate (bits/s)")
    st.subheader("Network traffic summary (synthetic)")
    flows = pd.DataFrame(demo["flow_telemetry"])
    if not flows.empty:
        st.metric("Flows", len(flows))
        st.metric("Packets", f'{int(flows["packet_count"].sum()):,}')
        st.metric("Bytes", f'{int(flows["byte_count"].sum()):,}')
        st.dataframe(flows, use_container_width=True)
    st.subheader("Cross-layer alerts")
    st.json({"evidence": assessment["evidence"], "scenario_scores": assessment["scenario_scores"]})
    st.subheader("Candidate/selected mitigation")
    st.json({
        "candidates": demo["candidate_mitigations"],
        "classical": demo["classical_solver"],
        "qaoa": demo["qaoa_solver"],
        "decision": demo["verifier"],
        "action": demo["mitigation"],
        "rollback": demo["rollback"],
    })
else:
    st.info("Run the mock demonstration to populate telemetry, risk, solver and audit views.")

if metrics_file.exists():
    st.subheader("Model metrics")
    metrics = pd.read_csv(metrics_file)
    st.dataframe(metrics, use_container_width=True)
    if "macro_f1" in metrics:
        st.line_chart(metrics.set_index("model")[["macro_f1"]])
    for image_path in sorted((root / "outputs" / "plots").glob("confusion-*.png")):
        st.image(str(image_path), caption=image_path.stem)
else:
    st.info("No dataset-backed model metrics exist yet. Run the training command after adding CSV data.")

prediction_paths = sorted((root / "outputs" / "metrics").glob("predictions-*.csv"))
if prediction_paths:
    st.subheader("QATNet/classical predictions and confidence")
    for prediction_path in prediction_paths:
        predictions = pd.read_csv(prediction_path)
        probability_columns = [column for column in predictions if column.startswith("probability_")]
        st.caption(prediction_path.stem)
        st.dataframe(predictions.head(100), use_container_width=True)
        if probability_columns:
            st.metric("Mean maximum confidence", f'{predictions[probability_columns].max(axis=1).mean():.3f}')

audit_path = root / "outputs" / "demo" / "audit.sqlite3"
if audit_path.exists():
    st.subheader("Audit log")
    with sqlite3.connect(audit_path) as connection:
        audit = pd.read_sql_query("SELECT timestamp, event_type, payload_json FROM audit_events ORDER BY id DESC", connection)
    if not audit.empty:
        st.dataframe(audit, use_container_width=True)
