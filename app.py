import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import acf, pacf
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

from modules.data_loader import (
    try_read_csv, detect_date_column, detect_value_column,
    get_candidate_columns, load_and_parse, ensure_freq,
)
from modules.preprocessor import (
    IMPUTE_METHODS, OUTLIER_METHODS, DENOISE_METHODS,
    recommend_impute_method, run_preprocessing,
    run_adf_test, render_stationarity_result,
)
from modules.forecaster import (
    MODEL_REGISTRY, NEEDS_STATIONARITY_CHECK,
    DECOMP_SEASONAL_MODELS, DECOMP_POLY_DEGREES,
    run_forecast, run_multi_forecast,
)
from modules.evaluator import (
    calc_metrics, build_comparison_table,
    get_best_model, render_metrics_cards,
)

st.set_page_config(page_title="시계열 예측 대시보드", layout="wide")
st.title("시계열 예측 대시보드")
st.caption("CSV 파일을 업로드하면 자동으로 분석·전처리·예측을 수행합니다.")

if "last_run" not in st.session_state:
    st.session_state["last_run"] = False

with st.sidebar:
    st.header("설정")
    uploaded = st.file_uploader("CSV 파일 업로드", type=["csv", "txt"])
    if uploaded is None:
        st.info("CSV 파일을 업로드하면 분석을 시작합니다.")

if uploaded is None:
    st.stop()

with st.sidebar:
    file_bytes = uploaded.read()
    try:
        raw_df, enc = try_read_csv(file_bytes)
        st.caption(f"인코딩: `{enc}` | {len(raw_df)}행 × {len(raw_df.columns)}열")
    except Exception as e:
        st.error(str(e)); st.stop()

    st.subheader("컬럼 설정")
    date_cands, value_cands = get_candidate_columns(raw_df)
    auto_date  = detect_date_column(raw_df)
    auto_value = detect_value_column(raw_df, auto_date)
    all_cols   = raw_df.columns.tolist()

    date_col = st.selectbox("날짜 컬럼",
        options=date_cands if date_cands else all_cols,
        index=(date_cands.index(auto_date) if auto_date and auto_date in date_cands else 0))
    value_col = st.selectbox("값 컬럼",
        options=value_cands if value_cands else all_cols,
        index=(value_cands.index(auto_value) if auto_value and auto_value in value_cands else 0))

    with st.spinner("데이터 파싱 중..."):
        parsed = load_and_parse(file_bytes, date_col, value_col)

    series_raw = parsed["series"]
    sp         = parsed["sp"]
    freq_label = parsed["freq_label"]
    st.caption(f"감지된 주기: **{freq_label}** | sp = {sp}")

    st.divider()
    st.subheader("전처리")

    rec_impute  = recommend_impute_method(series_raw, sp)
    impute_keys = list(IMPUTE_METHODS.keys())
    impute_label = st.radio("결측치 처리", options=impute_keys,
        index=(impute_keys.index(rec_impute) if rec_impute in impute_keys else 0))
    impute_window = 3
    if impute_label == "이동평균":
        impute_window = st.slider("이동평균 window", 3, 21, 5, step=2)

    st.markdown("**이상치 처리**")
    outlier_keys  = list(OUTLIER_METHODS.keys())
    outlier_label = st.radio("이상치 처리 방법", options=outlier_keys)
    outlier_window = st.slider("Hampel window 길이", 3, 15, 5, step=2)
    outlier_sigma  = st.slider("Hampel n_sigma", 2.0, 5.0, 3.0, step=0.5)

    with st.expander("고급 옵션 — 디노이징"):
        st.warning("대부분의 예측 모델은 내부 평활을 수행합니다. 센서·측정 노이즈가 명확한 경우에만 사용하세요.")
        denoise_keys  = list(DENOISE_METHODS.keys())
        denoise_label = st.radio("디노이징 방법", options=denoise_keys, index=0)
        denoise_window, denoise_alpha, denoise_cutoff = 5, 0.3, 0.1
        if denoise_label == "단순이동평균 (SMA)":
            denoise_window = st.slider("SMA window", 3, 21, 5, step=2)
        elif denoise_label == "지수이동평균 (EMA)":
            denoise_alpha = st.slider("EMA alpha", 0.1, 0.9, 0.3, step=0.1)
        elif denoise_label in ("FFT 저주파 통과 (LPF)", "FFT 고주파 통과 (HPF)"):
            denoise_cutoff = st.slider("차단 주파수 비율 (cutoff)", 0.01, 0.5, 0.1, step=0.01,
                help="낮을수록 더 강하게 필터링")

    st.divider()
    st.subheader("예측 파라미터")
    sp_input = st.number_input("계절 주기 (sp)", min_value=1, max_value=365, value=int(sp))

    st.markdown("**예측 시평 (horizon)**")
    horizon_max  = max(2, len(series_raw) // 2)
    horizon_mode = st.radio(
        "시평 설정 방식", ["상대적 시평", "절대적 시평"], horizontal=True,
        help="상대적: sp 배수로 지정 | 절대적: 예측할 시점 수를 직접 입력",
    )
    if horizon_mode == "상대적 시평":
        max_mult     = max(1, horizon_max // max(1, sp_input))
        default_mult = max(1, min(2, max_mult))
        sp_mult  = st.slider("계절 주기 배수", 1, max_mult, default_mult, format="%d sp")
        horizon  = min(sp_mult * sp_input, horizon_max)
        st.caption(f"→ **{horizon}** 시점 예측 (= {sp_mult} × sp {sp_input})")
    else:
        default_abs = min(sp_input, horizon_max)
        horizon     = st.slider("예측 시점 수", 1, horizon_max, default_abs)
        mult_ref    = horizon / max(1, sp_input)
        st.caption(f"→ **{horizon}** 시점 예측 (≈ {mult_ref:.1f} × sp {sp_input})")

    coverage = st.select_slider("신뢰구간 수준", [0.80, 0.90, 0.95], 0.90,
                                format_func=lambda x: f"{int(x*100)}%")

    st.divider()
    st.subheader("모델 선택")
    all_models     = list(MODEL_REGISTRY.keys())
    default_models = [m for m in ["Holt-Winter's 승법 (권장)", "STL Forecaster (권장)", "AutoARIMA"] if m in all_models]
    selected_models = st.multiselect("비교할 모델 선택", all_models, default=default_models)

    decomp_seasonal = "multiplicative"
    decomp_degree   = 1
    if "분해법 (Decomp)" in selected_models:
        with st.expander("분해법 (Decomp) 파라미터"):
            decomp_seasonal = st.radio(
                "계절성 모델", DECOMP_SEASONAL_MODELS,
                format_func=lambda x: "승법 (multiplicative)" if x == "multiplicative" else "가법 (additive)",
                horizontal=True,
            )
            decomp_degree = st.select_slider(
                "추세 다항식 차수", options=DECOMP_POLY_DEGREES, value=1,
                format_func=lambda x: f"{x}차 ({'선형' if x==1 else '이차' if x==2 else '삼차'})",
            )

    run_btn = st.button("예측 실행", type="primary", use_container_width=True)
    if run_btn:
        st.session_state["last_run"] = True

# ── 전처리 ───────────────────────────────────────────────────
with st.spinner("전처리 수행 중..."):
    prep = run_preprocessing(
        series=series_raw, sp=sp_input,
        impute_method=impute_label, impute_window=impute_window,
        outlier_method=outlier_label, outlier_window=outlier_window, outlier_sigma=outlier_sigma,
        denoise_method=denoise_label, denoise_window=denoise_window, denoise_alpha=denoise_alpha,
        denoise_cutoff=denoise_cutoff,
    )
series_clean = ensure_freq(prep["final"], parsed["freq"])

def to_ts(s):
    idx = s.index
    return idx.to_timestamp() if hasattr(idx, "to_timestamp") else idx

def _draw_acf_pacf(acf_v, pacf_v, conf, height=320, key_suffix=""):
    """ACF/PACF 서브플롯 공통 렌더링 헬퍼."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("자기상관함수 (ACF)", "편자기상관함수 (PACF)"),
        horizontal_spacing=0.08,
    )
    for ci, (arr, color) in enumerate([(acf_v, "#378ADD"), (pacf_v, "#7F77DD")], 1):
        lags = list(range(len(arr)))
        bar_colors = [color if abs(v) > conf else "rgba(255,255,255,0.15)" for v in arr]
        for i, v in enumerate(arr):
            fig.add_trace(go.Scatter(
                x=[i, i], y=[0, v], mode="lines",
                line=dict(color=color if abs(v) > conf else "rgba(255,255,255,0.2)", width=2),
                showlegend=False, hoverinfo="skip",
            ), row=1, col=ci)
        fig.add_trace(go.Scatter(
            x=lags, y=list(arr), mode="markers",
            marker=dict(color=bar_colors, size=6, line=dict(width=1, color="rgba(0,0,0,0.3)")),
            showlegend=False,
            hovertemplate="lag %{x}: <b>%{y:.3f}</b><extra></extra>",
        ), row=1, col=ci)
        fig.add_hrect(y0=-conf, y1=conf, fillcolor="rgba(226,75,74,0.07)", line_width=0, row=1, col=ci)
        fig.add_hline(y= conf, line_dash="dot", line_color="rgba(226,75,74,0.6)", line_width=1, row=1, col=ci)
        fig.add_hline(y=-conf, line_dash="dot", line_color="rgba(226,75,74,0.6)", line_width=1, row=1, col=ci)
        fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1, row=1, col=ci)
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, title="Lag"), xaxis2=dict(showgrid=False, title="Lag"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"), yaxis2=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"fig_acf_{key_suffix}")
    st.caption(f"점선: 95% 신뢰구간 ±{conf:.3f} | **진한 색** 막대: 통계적으로 유의한 자기상관")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["데이터 탐색", "전처리 결과", "예측 결과", "평가 대시보드", "모델 비교"])

# ── 탭 1: 데이터 탐색 ────────────────────────────────────────
with tab1:
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("데이터 수",    parsed["n_total"])
    c2.metric("결측치",       f"{parsed['n_missing']}개 ({parsed['missing_rate']*100:.1f}%)")
    c3.metric("감지된 주기",  freq_label)
    c4.metric("계절 주기 sp", sp_input)
    c5.metric("중복 타임스탬프", "있음" if parsed.get("has_duplicate") else "없음")
    st.divider()

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=to_ts(series_raw), y=series_raw.values,
        mode="lines", name="원본",
        line=dict(color="#378ADD", width=2),
        fill="tozeroy", fillcolor="rgba(55,138,221,0.08)",
        hovertemplate="%{x|%Y-%m-%d}<br>값: <b>%{y:.4f}</b><extra></extra>",
    ))
    fig1.update_layout(
        height=320, margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=True,
                   zerolinecolor="rgba(255,255,255,0.15)", zerolinewidth=1),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig1, use_container_width=True, key="fig_raw_series")

# ── 탭 2: 전처리 결과 ────────────────────────────────────────
with tab2:
    c1,c2,c3 = st.columns(3)
    c1.metric("결측치 처리", impute_label.split(" —")[0].split("(")[0].strip())
    c2.metric("이상치 탐지", prep["n_outliers"])
    c3.metric("디노이징", f"{denoise_label} ({prep['noise_reduction_rate']}% 감쇠)"
              if prep.get("denoise_applied") else "미적용")
    st.divider()

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=to_ts(series_raw), y=series_raw.values,
        mode="lines", name="원본", line=dict(color="#B4B2A9", width=1, dash="dot")))
    fig3.add_trace(go.Scatter(x=to_ts(series_clean), y=series_clean.values,
        mode="lines", name="전처리 후", line=dict(color="#1D9E75", width=2)))
    mask = prep.get("outlier_mask")
    if mask is not None and mask.sum() > 0:
        fig3.add_trace(go.Scatter(
            x=to_ts(series_raw)[mask.values], y=series_raw.values[mask.values],
            mode="markers", name="이상치", marker=dict(color="#E24B4A", size=8, symbol="x")))
    fig3.update_layout(height=320, margin=dict(l=0,r=0,t=20,b=0), hovermode="x unified",
                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig3, use_container_width=True, key="fig_preproc_compare")

    # ── ACF / PACF (전처리 후) ───────────────────────────────────
    st.divider()
    st.subheader("자기상관 구조")
    st.caption("전처리된 데이터 기준으로 ACF/PACF를 계산합니다.")
    vals = series_clean.dropna().values
    max_lags = min(40, len(vals)//2 - 1)
    if max_lags > 2:
        acf_v  = acf(vals, nlags=max_lags, fft=True)
        pacf_v = pacf(vals, nlags=max_lags)
        conf   = 1.96 / np.sqrt(len(vals))
        _draw_acf_pacf(acf_v, pacf_v, conf, height=320, key_suffix="clean")

    # ── 시계열 진단 검정 ─────────────────────────────────────────
    st.divider()
    st.subheader("시계열 진단 검정")
    st.caption("전처리된 데이터(series_clean) 기준으로 검정합니다.")

    diag_col1, diag_col2 = st.columns(2)

    with diag_col1:
        st.markdown("#### 정상성 검정 (ADF Test)")
        st.caption("귀무가설: 단위근이 존재한다 (비정상) → p < 0.05이면 정상")
        adf = run_adf_test(series_clean)

        if adf.get("error"):
            st.warning(adf["error"])
        else:
            is_stat = adf["is_stationary"]
            pval    = adf["pvalue"]
            if is_stat:
                st.success(f"**정상 시계열** (p = {pval})\n\n평균·분산이 시간에 따라 안정적입니다.")
            else:
                st.warning(f"**비정상 시계열** (p = {pval})\n\n추세 또는 분산이 시간에 따라 변하고 있습니다.")
            mc1, mc2 = st.columns(2)
            mc1.metric("ADF 통계량", adf["statistic"])
            mc2.metric("p-value",    pval)
            cv = adf["critical_values"]
            st.caption(f"임계값 — 1%: {cv.get('1%','-')} | 5%: {cv.get('5%','-')} | 10%: {cv.get('10%','-')}")

            if not is_stat:
                with st.expander("1차 차분 후 재검정"):
                    diff1 = series_clean.diff().dropna()
                    adf2  = run_adf_test(diff1)
                    if adf2.get("error"):
                        st.warning(adf2["error"])
                    else:
                        if adf2["is_stationary"]:
                            st.success(f"1차 차분 후 정상 (p = {adf2['pvalue']})\n\nd=1 차분으로 정상화 가능합니다.")
                        else:
                            diff2 = diff1.diff().dropna()
                            adf3  = run_adf_test(diff2)
                            if not adf3.get("error") and adf3["is_stationary"]:
                                st.success(f"2차 차분 후 정상 (p = {adf3['pvalue']})\n\nd=2 차분으로 정상화 가능합니다.")
                            else:
                                st.error("2차 차분 후에도 비정상입니다. 데이터 변환(로그 등)을 검토하세요.")
                        dc1, dc2 = st.columns(2)
                        dc1.metric("ADF 통계량 (차분 후)", adf2["statistic"])
                        dc2.metric("p-value (차분 후)",    adf2["pvalue"])

    with diag_col2:
        st.markdown("#### 자기상관성 검정 (Ljung-Box Test)")
        st.caption("귀무가설: 자기상관이 없다 → p < 0.05이면 유의미한 자기상관 존재")
        vals_clean = series_clean.dropna()
        max_lag    = min(20, len(vals_clean) // 2 - 1)
        if max_lag < 2:
            st.warning("데이터가 너무 짧아 검정 불가합니다.")
        else:
            lb_result = acorr_ljungbox(vals_clean.values, lags=max_lag, return_df=True)
            lb_pvals  = lb_result["lb_pvalue"]
            n_sig = int((lb_pvals < 0.05).sum())
            if n_sig > 0:
                st.info(f"**자기상관 존재** — {max_lag}개 lag 중 {n_sig}개에서 유의 (p < 0.05)\n\n"
                        "AR/MA 구조가 있을 가능성이 높습니다. ARIMA 계열 모델을 권장합니다.")
            else:
                st.success(f"**자기상관 없음** — 모든 lag에서 비유의 (p ≥ 0.05)\n\n"
                           "백색잡음에 가깝습니다. Naive 또는 평활 모델을 검토하세요.")
            fig_lb = go.Figure()
            fig_lb.add_trace(go.Bar(
                x=list(range(1, max_lag + 1)), y=lb_pvals.values,
                marker_color=["#E24B4A" if p < 0.05 else "#378ADD" for p in lb_pvals.values],
                name="p-value",
            ))
            fig_lb.add_hline(y=0.05, line_dash="dash", line_color="#E24B4A",
                             annotation_text="p=0.05", annotation_position="top right")
            fig_lb.update_layout(height=260, margin=dict(l=0,r=0,t=10,b=0),
                                 xaxis_title="Lag", yaxis_title="p-value",
                                 yaxis_range=[0,1], showlegend=False,
                                 plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_lb, use_container_width=True, key="fig_ljungbox")
            st.caption("빨간 막대: p < 0.05 (유의미한 자기상관) | 파란 막대: 비유의")

    # ── 차분 후 ACF/PACF (비정상 시) ─────────────────────────────
    if not adf.get("error") and not adf["is_stationary"]:
        diff1      = series_clean.diff().dropna()
        diff_vals  = diff1.dropna().values
        d_max_lags = min(20, len(diff_vals)//2 - 1)
        if d_max_lags > 2:
            d_acf_v  = acf(diff_vals, nlags=d_max_lags, fft=True)
            d_pacf_v = pacf(diff_vals, nlags=d_max_lags)
            d_conf   = 1.96 / np.sqrt(len(diff_vals))
            st.divider()
            st.markdown("##### 1차 차분 후 ACF / PACF")
            st.caption("차분 후 자기상관 구조를 확인하여 AR/MA 차수 결정에 참고하세요.")
            _draw_acf_pacf(d_acf_v, d_pacf_v, d_conf, height=260, key_suffix="diff")

# ── 탭 3: 예측 결과 ──────────────────────────────────────────
with tab3:
    if not selected_models:
        st.info("사이드바에서 모델을 선택하세요.")
    else:
        primary_model = st.selectbox("표시할 모델", selected_models, key="tab3_model")
        if primary_model in NEEDS_STATIONARITY_CHECK:
            with st.spinner("정상성 검정 중..."):
                adf_result = run_adf_test(series_clean)
            render_stationarity_result(adf_result)
            st.divider()

        if not st.session_state["last_run"]:
            st.info("사이드바에서 '예측 실행' 버튼을 눌러주세요.")
        else:
            with st.spinner(f"{primary_model} 학습 중..."):
                res = run_forecast(series_clean, primary_model, sp_input, horizon, coverage,
                                   decomp_seasonal=decomp_seasonal, decomp_degree=decomp_degree)
            if res.get("error"):
                st.error(f"예측 실패: {res['error']}")
            else:
                y_train, y_test, y_pred = res["y_train"], res["y_test"], res["y_pred"]
                _has_nonpositive = not (series_clean.dropna() > 0).all()
                if _has_nonpositive and "승법" in primary_model:
                    st.warning("데이터에 0 이하 값이 포함되어 있어 승법(multiplicative) 대신 "
                               "**가법(additive)** 계절성으로 자동 전환하여 학습했습니다.")
                if _has_nonpositive and primary_model == "분해법 (Decomp)" and decomp_seasonal == "multiplicative":
                    st.warning("데이터에 0 이하 값이 포함되어 있어 분해법의 계절성 모델을 "
                               "승법 → **가법(additive)** 으로 자동 전환하여 학습했습니다.")
                if res.get("model_info"):
                    info  = res["model_info"]
                    order = info.get("order", "?")
                    st.info(f"AutoARIMA — order: `{order}`, seasonal: `{info.get('seasonal_order','?')}`, "
                            f"AIC: `{info.get('aic','N/A')}`")
                    if order and len(order) >= 2 and order[1] > 0:
                        st.caption(f"d={order[1]} 적용: 비정상 데이터를 {order[1]}번 차분하여 학습했습니다.")

                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(x=to_ts(y_train), y=y_train.values,
                    name="학습 데이터", line=dict(color="#378ADD", width=1.5)))
                fig4.add_trace(go.Scatter(x=to_ts(y_test), y=y_test.values,
                    name="실제값", line=dict(color="#888780", width=1.5, dash="dot")))
                fig4.add_trace(go.Scatter(x=to_ts(y_pred), y=y_pred.values,
                    name="예측값", line=dict(color="#E24B4A", width=2)))
                if res.get("y_interval") is not None:
                    try:
                        yi = res["y_interval"]
                        px = list(to_ts(y_pred))
                        fig4.add_trace(go.Scatter(
                            x=px + px[::-1],
                            y=list(yi.iloc[:,1].values) + list(yi.iloc[:,0].values[::-1]),
                            fill="toself", fillcolor="rgba(226,75,74,0.15)",
                            line=dict(color="rgba(0,0,0,0)"),
                            name=f"신뢰구간 {int(coverage*100)}%"))
                    except Exception:
                        pass
                if res.get("y_future") is not None:
                    fig4.add_trace(go.Scatter(x=to_ts(res["y_future"]), y=res["y_future"].values,
                        name="미래 예측", line=dict(color="#EF9F27", width=2, dash="dash")))
                fig4.update_layout(height=380, margin=dict(l=0,r=0,t=20,b=0), hovermode="x unified",
                                   plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig4, use_container_width=True, key="fig_forecast")
                render_metrics_cards(calc_metrics(y_test, y_pred, y_train))

# ── 탭 4: 평가 대시보드 ──────────────────────────────────────
with tab4:
    if not st.session_state["last_run"] or not selected_models:
        st.info("예측을 먼저 실행하세요.")
    else:
        primary_model4 = st.selectbox("평가할 모델", selected_models, key="tab4_model")
        with st.spinner("평가 중..."):
            res4 = run_forecast(series_clean, primary_model4, sp_input, horizon, coverage,
                                decomp_seasonal=decomp_seasonal, decomp_degree=decomp_degree)
        if res4.get("error") or res4.get("y_pred") is None:
            st.error("평가할 예측 결과가 없습니다.")
        else:
            y_test4, y_pred4, y_tr4 = res4["y_test"], res4["y_pred"], res4["y_train"]
            render_metrics_cards(calc_metrics(y_test4, y_pred4, y_tr4))
            st.divider()
            residuals = y_test4 - y_pred4
            res_vals  = residuals.dropna().values

            from statsmodels.tsa.stattools import acf as acf_fn
            from scipy.stats import norm as sp_norm

            fig5 = make_subplots(
                rows=2, cols=2,
                subplot_titles=("잔차 플롯", "잔차 분포", "잔차 ACF", "Q-Q 플롯"),
                vertical_spacing=0.14, horizontal_spacing=0.10,
            )
            # 잔차 플롯
            fig5.add_trace(go.Bar(
                x=to_ts(residuals), y=residuals.values,
                marker_color=["#1D9E75" if v >= 0 else "#E24B4A" for v in residuals.values],
                marker_line_width=0, name="잔차",
                hovertemplate="%{x|%Y-%m-%d}<br>잔차: <b>%{y:.4f}</b><extra></extra>",
            ), row=1, col=1)
            fig5.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1.5, line_dash="dash", row=1, col=1)
            # 히스토그램 + 정규분포 곡선
            fig5.add_trace(go.Histogram(x=res_vals, nbinsx=15,
                marker_color="#378ADD", marker_line_color="rgba(0,0,0,0.3)",
                marker_line_width=1, opacity=0.85, name="분포",
            ), row=1, col=2)
            x_range  = np.linspace(res_vals.min(), res_vals.max(), 100)
            mu, sigma = res_vals.mean(), res_vals.std()
            bin_width = (res_vals.max() - res_vals.min()) / 15
            pdf_vals  = sp_norm.pdf(x_range, mu, sigma) * len(res_vals) * bin_width
            fig5.add_trace(go.Scatter(x=x_range, y=pdf_vals, mode="lines",
                line=dict(color="#EF9F27", width=2), name="정규분포", hoverinfo="skip",
            ), row=1, col=2)
            # 잔차 ACF
            if len(res_vals) > 5:
                r_nlags = min(20, len(res_vals)//2 - 1)
                r_acf   = acf_fn(res_vals, nlags=r_nlags, fft=True)
                conf_r  = 1.96 / np.sqrt(len(res_vals))
                for i, v in enumerate(r_acf):
                    fig5.add_trace(go.Scatter(x=[i,i], y=[0,v], mode="lines",
                        line=dict(color="#1D9E75" if abs(v) > conf_r else "rgba(255,255,255,0.2)", width=2),
                        showlegend=False, hoverinfo="skip",
                    ), row=2, col=1)
                fig5.add_trace(go.Scatter(
                    x=list(range(len(r_acf))), y=list(r_acf), mode="markers",
                    marker=dict(color=["#1D9E75" if abs(v) > conf_r else "rgba(255,255,255,0.3)" for v in r_acf], size=6),
                    showlegend=False, hovertemplate="lag %{x}: <b>%{y:.3f}</b><extra></extra>",
                ), row=2, col=1)
                fig5.add_hrect(y0=-conf_r, y1=conf_r, fillcolor="rgba(226,75,74,0.07)", line_width=0, row=2, col=1)
                fig5.add_hline(y= conf_r, line_dash="dot", line_color="rgba(226,75,74,0.5)", line_width=1, row=2, col=1)
                fig5.add_hline(y=-conf_r, line_dash="dot", line_color="rgba(226,75,74,0.5)", line_width=1, row=2, col=1)
                fig5.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_width=1, row=2, col=1)
            # Q-Q 플롯
            (osm, osr), (slope, intercept, _) = stats.probplot(res_vals)
            fig5.add_trace(go.Scatter(x=osm, y=osr, mode="markers",
                marker=dict(color="#7F77DD", size=7, line=dict(width=1, color="rgba(0,0,0,0.3)")),
                name="Q-Q", hovertemplate="이론: %{x:.3f}<br>실제: <b>%{y:.3f}</b><extra></extra>",
            ), row=2, col=2)
            fig5.add_trace(go.Scatter(x=[min(osm),max(osm)],
                y=[slope*min(osm)+intercept, slope*max(osm)+intercept],
                mode="lines", line=dict(color="#E24B4A", dash="dash", width=1.5),
                showlegend=False, hoverinfo="skip",
            ), row=2, col=2)
            fig5.update_layout(height=580, margin=dict(l=0,r=0,t=40,b=0), showlegend=False,
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", bargap=0.05)
            for axis in ["xaxis","xaxis2","xaxis3","xaxis4"]:
                fig5.update_layout(**{axis: dict(showgrid=False, zeroline=False)})
            for axis in ["yaxis","yaxis2","yaxis3","yaxis4"]:
                fig5.update_layout(**{axis: dict(gridcolor="rgba(255,255,255,0.05)")})
            st.plotly_chart(fig5, use_container_width=True, key="fig_residuals")
            st.caption("잔차 플롯: **초록** = 과소예측, **빨강** = 과대예측 | "
                       "잔차 ACF: **진한 초록** = 유의한 자기상관 | "
                       "Q-Q 플롯: 점이 선에 가까울수록 정규성 만족")

# ── 탭 5: 모델 비교 ──────────────────────────────────────────
with tab5:
    if not selected_models:
        st.info("사이드바에서 모델을 선택하세요.")
    elif not st.session_state["last_run"]:
        st.info("예측을 먼저 실행하세요.")
    else:
        with st.spinner("모든 모델 학습 중..."):
            all_results = run_multi_forecast(series_clean, selected_models, sp_input, horizon,
                                             decomp_seasonal=decomp_seasonal, decomp_degree=decomp_degree)
        comp_df   = build_comparison_table(all_results, series_clean)
        best_name = get_best_model(comp_df, "MAPE(%)")
        if best_name:
            st.success(f"MAPE 기준 최적 모델: **{best_name}**")

        metric_cols       = [c for c in ["MAE","RMSE","MAPE(%)","SMAPE(%)","MdAE","MASE","RSFE","TS"] if c in comp_df.columns]
        valid_metric_cols = [c for c in metric_cols if comp_df[c].notna().any()]

        def style_comparison(df):
            styled = pd.DataFrame("", index=df.index, columns=df.columns)
            if "오류" in df.columns:
                for idx in df.index:
                    if pd.notna(df.loc[idx, "오류"]) and df.loc[idx, "오류"] not in (None, ""):
                        for col in df.columns:
                            styled.loc[idx, col] = "color: #E24B4A; font-style: italic;"
            abs_neutral = {"RSFE", "TS"}
            for col in valid_metric_cols:
                col_data = df[col].copy()
                if col in abs_neutral:
                    col_data = col_data.abs()
                valid = col_data.dropna()
                if len(valid) < 2:
                    continue
                col_min, col_max = valid.min(), valid.max()
                rng = col_max - col_min if col_max != col_min else 1
                for idx in df.index:
                    val = col_data.loc[idx]
                    if pd.isna(val):
                        continue
                    ratio = (val - col_min) / rng
                    if ratio < 0.2:
                        styled.loc[idx, col] = "background-color: #0d4f3c; color: #5DCAA5; font-weight: 600;"
                    elif ratio < 0.4:
                        styled.loc[idx, col] = "background-color: #1a3a2a; color: #8ed4b8;"
                    elif ratio > 0.8:
                        styled.loc[idx, col] = "background-color: #4a1a1a; color: #E24B4A;"
                    elif ratio > 0.6:
                        styled.loc[idx, col] = "background-color: #3a2020; color: #e87070;"
            return styled

        st.dataframe(
            comp_df.style.apply(style_comparison, axis=None).format(precision=4, na_rep="—"),
            use_container_width=True,
        )
        st.divider()
        COLORS = ["#E24B4A","#1D9E75","#378ADD","#EF9F27","#7F77DD","#D4537E"]
        fig6 = go.Figure()
        first = True
        for i, (label, ri) in enumerate(all_results.items()):
            if ri.get("error") or ri.get("y_pred") is None:
                continue
            if first:
                fig6.add_trace(go.Scatter(x=to_ts(ri["y_train"]), y=ri["y_train"].values,
                    name="학습 데이터", line=dict(color="#B4B2A9", width=1)))
                fig6.add_trace(go.Scatter(x=to_ts(ri["y_test"]), y=ri["y_test"].values,
                    name="실제값", line=dict(color="#444441", width=2, dash="dot")))
                first = False
            fig6.add_trace(go.Scatter(x=to_ts(ri["y_pred"]), y=ri["y_pred"].values,
                name=label, line=dict(color=COLORS[i % len(COLORS)], width=1.5)))
        fig6.update_layout(height=380, margin=dict(l=0,r=0,t=10,b=0), hovermode="x unified",
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig6, use_container_width=True, key="fig_model_compare")
        st.divider()
        if not comp_df.empty:
            st.download_button("비교 결과 CSV 다운로드",
                data=comp_df.to_csv(encoding="utf-8-sig"),
                file_name="model_comparison.csv", mime="text/csv")