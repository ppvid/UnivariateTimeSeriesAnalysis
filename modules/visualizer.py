import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── 공통 색상 팔레트 ──────────────────────────────────────────
COLOR = {
    "original":  "#378ADD",
    "train":     "#378ADD",
    "test":      "#EF9F27",
    "pred":      "#1D9E75",
    "future":    "#7F77DD",
    "outlier":   "#E24B4A",
    "denoised":  "#D85A30",
    "residual":  "#888780",
    "ci_fill":   "rgba(29, 158, 117, 0.15)",
}

LAYOUT_BASE = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(size=12),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    hovermode="x unified",
)


def _base_layout(**kwargs):
    d = LAYOUT_BASE.copy()
    d.update(kwargs)
    return d


# ── 원본 시계열 플롯 ──────────────────────────────────────────
def plot_raw_series(series: pd.Series, title: str = "원본 시계열") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index.astype(str),
        y=series.values,
        mode="lines",
        name="원본",
        line=dict(color=COLOR["original"], width=1.5),
    ))
    fig.update_layout(**_base_layout(title=title))
    return fig


# ── train/test 분할 시각화 ────────────────────────────────────
def plot_train_test_split(
    y_train: pd.Series,
    y_test: pd.Series,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_train.index.astype(str), y=y_train.values,
        mode="lines", name="학습 데이터",
        line=dict(color=COLOR["train"], width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=y_test.index.astype(str), y=y_test.values,
        mode="lines", name="평가 데이터",
        line=dict(color=COLOR["test"], width=1.5),
    ))
    fig.update_layout(**_base_layout(title="학습 / 평가 데이터 분할"))
    return fig


# ── 전처리 비교 플롯 (원본 vs 처리 후) ───────────────────────
def plot_preprocessing_comparison(
    original: pd.Series,
    processed: pd.Series,
    outlier_mask: pd.Series | None = None,
    title: str = "전처리 결과 비교",
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=original.index.astype(str), y=original.values,
        mode="lines", name="원본",
        line=dict(color=COLOR["original"], width=1, dash="dot"),
        opacity=0.6,
    ))
    fig.add_trace(go.Scatter(
        x=processed.index.astype(str), y=processed.values,
        mode="lines", name="처리 후",
        line=dict(color=COLOR["pred"], width=1.8),
    ))

    if outlier_mask is not None and outlier_mask.sum() > 0:
        outlier_idx = original.index[outlier_mask]
        fig.add_trace(go.Scatter(
            x=outlier_idx.astype(str),
            y=original[outlier_mask].values,
            mode="markers", name="탐지된 이상치",
            marker=dict(color=COLOR["outlier"], size=8, symbol="x"),
        ))

    fig.update_layout(**_base_layout(title=title))
    return fig


# ── 디노이징 비교 플롯 ────────────────────────────────────────
def plot_denoising_comparison(
    original: pd.Series,
    denoised: pd.Series,
    noise_rate: float,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=original.index.astype(str), y=original.values,
        mode="lines", name="디노이징 전",
        line=dict(color=COLOR["original"], width=1, dash="dot"),
        opacity=0.5,
    ))
    fig.add_trace(go.Scatter(
        x=denoised.index.astype(str), y=denoised.values,
        mode="lines", name=f"디노이징 후 (감쇠율 {noise_rate:.1f}%)",
        line=dict(color=COLOR["denoised"], width=2),
    ))
    fig.update_layout(**_base_layout(title="디노이징 비교"))
    return fig


# ── 예측 결과 플롯 ────────────────────────────────────────────
def plot_forecast(
    y_train: pd.Series,
    y_test: pd.Series,
    y_pred: pd.Series,
    model_label: str = "예측",
    y_future: pd.Series | None = None,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=y_train.index.astype(str), y=y_train.values,
        mode="lines", name="학습 데이터",
        line=dict(color=COLOR["train"], width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=y_test.index.astype(str), y=y_test.values,
        mode="lines", name="실제값 (평가)",
        line=dict(color=COLOR["test"], width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=y_pred.index.astype(str), y=y_pred.values,
        mode="lines", name=f"예측 ({model_label})",
        line=dict(color=COLOR["pred"], width=2, dash="dash"),
    ))

    if y_future is not None:
        fig.add_trace(go.Scatter(
            x=y_future.index.astype(str), y=y_future.values,
            mode="lines", name="미래 예측",
            line=dict(color=COLOR["future"], width=2, dash="dot"),
        ))

    fig.update_layout(**_base_layout(title=f"예측 결과 — {model_label}"))
    return fig


# ── 잔차 분석 플롯 (2×2) ─────────────────────────────────────
def plot_residuals(
    residuals: pd.Series,
    model_label: str = "",
) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["잔차 플롯", "잔차 분포 (히스토그램)", "ACF (자기상관)", "Q-Q 플롯"],
    )

    # 잔차 시계열
    fig.add_trace(go.Scatter(
        x=residuals.index.astype(str), y=residuals.values,
        mode="lines+markers", name="잔차",
        line=dict(color=COLOR["residual"], width=1),
        marker=dict(size=4),
    ), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)

    # 히스토그램
    fig.add_trace(go.Histogram(
        x=residuals.values, name="분포",
        marker_color=COLOR["residual"], opacity=0.7,
        nbinsx=20,
    ), row=1, col=2)

    # ACF (간이 계산)
    n = len(residuals)
    max_lags = min(20, n // 2)
    acf_vals = [residuals.autocorr(lag=k) for k in range(1, max_lags + 1)]
    conf = 1.96 / np.sqrt(n)

    fig.add_trace(go.Bar(
        x=list(range(1, max_lags + 1)),
        y=acf_vals, name="ACF",
        marker_color=COLOR["residual"],
    ), row=2, col=1)
    fig.add_hline(y=conf,  line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
    fig.add_hline(y=-conf, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)

    # Q-Q 플롯
    sorted_res = np.sort(residuals.dropna().values)
    n_pts = len(sorted_res)
    theoretical = np.array([
        np.percentile(np.random.normal(0, 1, 1000), (i / n_pts) * 100)
        for i in range(1, n_pts + 1)
    ])
    fig.add_trace(go.Scatter(
        x=theoretical, y=sorted_res,
        mode="markers", name="Q-Q",
        marker=dict(color=COLOR["residual"], size=4),
    ), row=2, col=2)
    mn = min(theoretical.min(), sorted_res.min())
    mx = max(theoretical.max(), sorted_res.max())
    fig.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx],
        mode="lines", name="기준선",
        line=dict(color="gray", dash="dash"),
    ), row=2, col=2)

    fig.update_layout(
        **_base_layout(
            title=f"잔차 분석 — {model_label}",
            height=520,
            showlegend=False,
        )
    )
    return fig


# ── 모델 비교 막대 차트 ───────────────────────────────────────
def plot_model_comparison(compare_df: pd.DataFrame, metric: str = "MAPE(%)") -> go.Figure:
    df = compare_df[compare_df["상태"] == "완료"].copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=[metric]).sort_values(metric)

    colors = [COLOR["pred"] if i == 0 else COLOR["residual"] for i in range(len(df))]

    fig = go.Figure(go.Bar(
        x=df["모델"],
        y=df[metric],
        marker_color=colors,
        text=df[metric].round(3).astype(str),
        textposition="outside",
    ))
    fig.update_layout(**_base_layout(
        title=f"모델 비교 — {metric}",
        yaxis_title=metric,
        xaxis_title="",
    ))
    return fig


# ── ACF/PACF 플롯 (탐색 탭용) ────────────────────────────────
def plot_acf_pacf(series: pd.Series, max_lags: int = 30) -> go.Figure:
    clean = series.dropna()
    n = len(clean)
    max_lags = min(max_lags, n // 2 - 1)
    conf = 1.96 / np.sqrt(n)

    acf_vals = [clean.autocorr(lag=k) for k in range(1, max_lags + 1)]
    pacf_vals = []
    try:
        from statsmodels.tsa.stattools import pacf as sm_pacf
        pacf_vals = sm_pacf(clean.values, nlags=max_lags, method="ols")[1:]
    except Exception:
        pacf_vals = [0.0] * max_lags

    lags = list(range(1, max_lags + 1))
    fig = make_subplots(rows=1, cols=2, subplot_titles=["ACF (자기상관함수)", "PACF (편자기상관함수)"])

    for col_idx, (vals, name) in enumerate([(acf_vals, "ACF"), (pacf_vals, "PACF")], 1):
        fig.add_trace(go.Bar(x=lags, y=vals, name=name,
                             marker_color=COLOR["original"]), row=1, col=col_idx)
        fig.add_hline(y=conf,  line_dash="dash", line_color="gray", opacity=0.5, row=1, col=col_idx)
        fig.add_hline(y=-conf, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=col_idx)

    fig.update_layout(**_base_layout(title="ACF / PACF", height=320, showlegend=False))
    return fig
