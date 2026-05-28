import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from src.data_loader import available_plants, load_plant
from src.model import predict, train
from src.preprocessing import compute_features
from src.visualization import (
    anomaly_heatmap,
    anomaly_rate_bar,
    anomaly_timeline,
    feature_scatter,
    hourly_profile,
    inverter_timeseries,
    model_comparison_chart,
    score_distribution,
)

st.set_page_config(page_title="Solar Anomaly Detection", layout="wide")
st.title("Solar Inverter Anomaly Detection")

# ── Sidebar ───────────────────────────────────────────────────────────────────
plants = available_plants()
if not plants:
    st.warning("No CSV files found in data/raw/.")
    st.stop()

plant_id = st.sidebar.selectbox("Plant", plants, format_func=lambda x: f"Plant {x}")
contamination = st.sidebar.slider(
    "Contamination", 0.01, 0.20, 0.05, 0.01,
    help="Fraction estimée de points anormaux",
)
MODEL_LABELS = {
    "isolation_forest": "Isolation Forest",
    "lof": "LOF",
    "zscore": "Z-score",
}
model_type = st.sidebar.selectbox(
    "Modèle",
    list(MODEL_LABELS.keys()),
    format_func=MODEL_LABELS.get,
)


# ── Pipeline (caché) ──────────────────────────────────────────────────────────
@st.cache_data
def load_features(pid: int) -> pd.DataFrame:
    generation, weather = load_plant(pid)
    return compute_features(generation, weather)


@st.cache_data
def run_detection(pid: int, contam: float, mtype: str) -> pd.DataFrame:
    df = load_features(pid)
    return predict(train(df, contamination=contam, model_type=mtype), df)


@st.cache_data
def run_all_models(pid: int, contam: float) -> dict[str, pd.DataFrame]:
    df = load_features(pid)
    return {
        label: predict(train(df, contamination=contam, model_type=mtype), df)
        for mtype, label in MODEL_LABELS.items()
    }


df = run_detection(plant_id, contamination, model_type)

# ── Métriques globales ────────────────────────────────────────────────────────
n_total = len(df)
n_anomalies = int(df["anomaly"].sum())
n_inverters = df["SOURCE_KEY"].nunique()
worst_inv = df.groupby("SOURCE_KEY")["anomaly"].mean().idxmax()
worst_rate = df.groupby("SOURCE_KEY")["anomaly"].mean().max()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Mesures analysées", f"{n_total:,}")
col2.metric("Anomalies détectées", f"{n_anomalies:,}", f"{n_anomalies/n_total:.1%}")
col3.metric("Onduleurs", n_inverters)
col4.metric("Onduleur le + anormal", worst_inv, f"{worst_rate:.1%}")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_overview, tab_inverter, tab_analysis, tab_compare = st.tabs(
    ["Vue d'ensemble", "Détail onduleur", "Analyse du modèle", "Comparaison modèles"]
)

with tab_overview:
    st.subheader("Taux d'anomalie par onduleur")
    st.plotly_chart(anomaly_rate_bar(df, contamination), use_container_width=True)

    st.subheader("Heatmap anomalies — onduleur × jour")
    st.plotly_chart(anomaly_heatmap(df), use_container_width=True)

    st.subheader("Évolution temporelle")
    st.plotly_chart(anomaly_timeline(df), use_container_width=True)

with tab_inverter:
    inverter = st.selectbox(
        "Onduleur",
        sorted(df["SOURCE_KEY"].unique()),
        index=sorted(df["SOURCE_KEY"].unique()).index(worst_inv),
    )

    st.plotly_chart(inverter_timeseries(df, inverter), use_container_width=True)

    st.subheader("Anomalies dans l'espace des features")
    col_a, col_b = st.columns(2)
    inv_df = df[df["SOURCE_KEY"] == inverter]
    with col_a:
        st.plotly_chart(
            feature_scatter(inv_df, "DC_POWER", "perf_ratio"),
            use_container_width=True,
        )
    with col_b:
        st.plotly_chart(
            feature_scatter(inv_df, "DC_POWER", "power_deviation"),
            use_container_width=True,
        )

    st.subheader("Statistiques")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("DC_POWER moyen", f"{inv_df['DC_POWER'].mean():.0f} W")
    c2.metric("Perf ratio moyen", f"{inv_df['perf_ratio'].mean():.2f}")
    c3.metric("Rendement moyen", f"{inv_df['efficiency'].mean():.3f}")
    c4.metric("Taux anomalie", f"{inv_df['anomaly'].mean():.1%}")

with tab_analysis:
    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(score_distribution(df), use_container_width=True)
    with col_right:
        st.plotly_chart(hourly_profile(df, contamination), use_container_width=True)

    st.subheader("Scatter global — tous onduleurs")
    x_feat = st.selectbox(
        "Feature X", ["DC_POWER", "perf_ratio", "power_deviation", "efficiency"], index=0
    )
    y_feat = st.selectbox(
        "Feature Y", ["perf_ratio", "DC_POWER", "power_deviation", "efficiency"], index=0
    )
    st.plotly_chart(feature_scatter(df, x_feat, y_feat), use_container_width=True)

with tab_compare:
    st.info(
        "Compare les 3 modèles avec la même contamination. "
        "Les onduleurs signalés par plusieurs modèles sont les plus fiables."
    )
    with st.spinner("Entraînement des 3 modèles…"):
        results = run_all_models(plant_id, contamination)

    st.plotly_chart(model_comparison_chart(results), use_container_width=True)

    # Tableau consensus
    st.subheader("Tableau de consensus")
    consensus = pd.DataFrame(
        {label: res.groupby("SOURCE_KEY")["anomaly"].mean() for label, res in results.items()}
    )
    consensus.columns = [f"Taux {c}" for c in consensus.columns]
    consensus["Consensus (nb modèles > seuil)"] = (
        (consensus > contamination * 1.5).sum(axis=1)
    )
    consensus = consensus.sort_values("Consensus (nb modèles > seuil)", ascending=False)
    st.dataframe(
        consensus.style
        .background_gradient(subset=[c for c in consensus.columns if "Taux" in c], cmap="Reds")
        .format({c: "{:.1%}" for c in consensus.columns if "Taux" in c}),
        use_container_width=True,
    )
