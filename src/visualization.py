import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_NORMAL_COLOR = "#4C9BE8"
_ANOMALY_COLOR = "#E84C4C"


def anomaly_rate_bar(df: pd.DataFrame, contamination: float = 0.05) -> go.Figure:
    rate = (
        df.groupby("SOURCE_KEY")["anomaly"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"anomaly": "anomaly_rate"})
    )
    rate["color"] = rate["anomaly_rate"].apply(
        lambda r: _ANOMALY_COLOR if r > contamination * 1.5 else _NORMAL_COLOR
    )

    fig = go.Figure(
        go.Bar(
            x=rate["SOURCE_KEY"],
            y=rate["anomaly_rate"],
            marker_color=rate["color"],
            hovertemplate="%{x}<br>Taux : %{y:.1%}<extra></extra>",
        )
    )
    fig.add_hline(
        y=contamination,
        line_dash="dash",
        line_color="black",
        annotation_text=f"contamination = {contamination:.0%}",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Taux d'anomalie par onduleur",
        xaxis_title="Onduleur",
        yaxis_title="Taux d'anomalie",
        yaxis_tickformat=".0%",
        height=400,
    )
    return fig


def anomaly_heatmap(df: pd.DataFrame) -> go.Figure:
    rate = (
        df.groupby(["SOURCE_KEY", "date"])["anomaly"]
        .mean()
        .reset_index()
        .rename(columns={"anomaly": "rate"})
    )
    rate["date"] = rate["date"].astype(str)

    inv_order = (
        df.groupby("SOURCE_KEY")["anomaly"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    pivot = (
        rate.pivot(index="SOURCE_KEY", columns="date", values="rate")
        .reindex(inv_order)
    )

    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="Reds",
            zmin=0,
            zmax=1,
            colorbar=dict(title="Taux"),
            hovertemplate="Onduleur: %{y}<br>Date: %{x}<br>Taux: %{z:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Taux d'anomalie par onduleur et par jour",
        xaxis_title="Date",
        yaxis_title="Onduleur",
        height=500,
    )
    return fig


def inverter_timeseries(df: pd.DataFrame, inverter: str) -> go.Figure:
    sub = df[df["SOURCE_KEY"] == inverter].sort_values("DATE_TIME")
    normal = sub[sub["anomaly"] == 0]
    anomaly = sub[sub["anomaly"] == 1]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=normal["DATE_TIME"],
            y=normal["DC_POWER"],
            mode="markers",
            marker=dict(size=3, color=_NORMAL_COLOR, opacity=0.5),
            name="Normal",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=anomaly["DATE_TIME"],
            y=anomaly["DC_POWER"],
            mode="markers",
            marker=dict(size=7, color=_ANOMALY_COLOR, symbol="x", opacity=0.9),
            name="Anomalie",
        )
    )
    fig.update_layout(
        title=f"DC_POWER — {inverter}",
        xaxis_title="Date",
        yaxis_title="DC_POWER (W)",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def score_distribution(df: pd.DataFrame) -> go.Figure:
    threshold = df.loc[df["anomaly"] == 1, "anomaly_score"].min()
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=df.loc[df["anomaly"] == 0, "anomaly_score"],
            nbinsx=100,
            histnorm="probability density",
            name="Normal",
            marker_color=_NORMAL_COLOR,
            opacity=0.7,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=df.loc[df["anomaly"] == 1, "anomaly_score"],
            nbinsx=100,
            histnorm="probability density",
            name="Anomalie",
            marker_color=_ANOMALY_COLOR,
            opacity=0.7,
        )
    )
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="black",
        annotation_text=f"seuil {threshold:.3f}",
    )
    fig.update_layout(
        title="Distribution des scores d'anomalie",
        xaxis_title="Anomaly Score",
        yaxis_title="Densité",
        barmode="overlay",
        height=380,
    )
    return fig


def feature_scatter(df: pd.DataFrame, x_col: str, y_col: str) -> go.Figure:
    color_map = {0: _NORMAL_COLOR, 1: _ANOMALY_COLOR}
    label_map = {0: "Normal", 1: "Anomalie"}

    fig = go.Figure()
    for val in [0, 1]:
        sub = df[df["anomaly"] == val]
        fig.add_trace(
            go.Scatter(
                x=sub[x_col],
                y=sub[y_col],
                mode="markers",
                marker=dict(
                    size=3 if val == 0 else 5,
                    color=color_map[val],
                    opacity=0.15 if val == 0 else 0.5,
                ),
                name=label_map[val],
            )
        )
    fig.update_layout(
        title=f"{x_col} vs {y_col}",
        xaxis_title=x_col,
        yaxis_title=y_col,
        height=380,
    )
    return fig


def hourly_profile(df: pd.DataFrame, contamination: float = 0.05) -> go.Figure:
    high_keys = (
        df.groupby("SOURCE_KEY")["anomaly"]
        .mean()
        .pipe(lambda s: s[s > contamination * 1.5])
        .index
    )
    low_keys = (
        df.groupby("SOURCE_KEY")["anomaly"]
        .mean()
        .pipe(lambda s: s[s <= contamination * 1.5])
        .index
    )

    h_high = df[df["SOURCE_KEY"].isin(high_keys)].groupby("hour")["DC_POWER"].mean()
    h_low = df[df["SOURCE_KEY"].isin(low_keys)].groupby("hour")["DC_POWER"].mean()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=h_low.index, y=h_low.values,
            mode="lines+markers", name=f"Normaux ({len(low_keys)})",
            line=dict(color=_NORMAL_COLOR),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=h_high.index, y=h_high.values,
            mode="lines+markers", name=f"Anormaux ({len(high_keys)})",
            line=dict(color=_ANOMALY_COLOR),
        )
    )
    fig.add_traces(
        go.Scatter(
            x=list(h_low.index) + list(h_low.index[::-1]),
            y=list(h_high.values) + list(h_low.values[::-1]),
            fill="toself",
            fillcolor="rgba(232,76,76,0.1)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title="Profil journalier moyen : normaux vs anormaux",
        xaxis_title="Heure",
        yaxis_title="DC_POWER moyen (W)",
        height=380,
    )
    return fig


def model_comparison_chart(results: dict[str, pd.DataFrame]) -> go.Figure:
    """Grouped bar chart : taux d'anomalie par onduleur pour chaque modèle."""
    colors = {"Isolation Forest": "#4C9BE8", "LOF": "#F4A261", "Z-score": "#2A9D8F"}

    all_keys = sorted(
        next(iter(results.values()))["SOURCE_KEY"].unique()
    )

    fig = go.Figure()
    for label, df in results.items():
        rate = df.groupby("SOURCE_KEY")["anomaly"].mean().reindex(all_keys)
        fig.add_trace(
            go.Bar(
                name=label,
                x=all_keys,
                y=rate.values,
                marker_color=colors.get(label, "#888"),
                hovertemplate=f"<b>{label}</b><br>%{{x}}<br>Taux : %{{y:.1%}}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Taux d'anomalie par onduleur — comparaison des modèles",
        xaxis_title="Onduleur",
        yaxis_title="Taux d'anomalie",
        yaxis_tickformat=".0%",
        barmode="group",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def anomaly_timeline(df: pd.DataFrame) -> go.Figure:
    ts = (
        df.groupby("DATE_TIME")
        .agg(score_mean=("anomaly_score", "mean"), n_anomalies=("anomaly", "sum"))
        .reset_index()
    )

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("Score moyen", "Nb onduleurs en anomalie"),
        vertical_spacing=0.08,
    )
    fig.add_trace(
        go.Scatter(
            x=ts["DATE_TIME"], y=ts["score_mean"],
            line=dict(width=1, color="darkorange"), name="Score moyen",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ts["DATE_TIME"], y=ts["n_anomalies"],
            fill="tozeroy", line=dict(width=0.5, color=_ANOMALY_COLOR),
            fillcolor="rgba(232,76,76,0.3)", name="Nb anomalies",
        ),
        row=2, col=1,
    )
    fig.update_layout(title="Évolution temporelle des anomalies", height=450, showlegend=False)
    return fig
