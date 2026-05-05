import pandas as pd
import numpy as np
import streamlit as st
from statsmodels.tsa.stattools import adfuller

from sktime.transformations.series.impute import Imputer
from sktime.forecasting.exp_smoothing import ExponentialSmoothing as SktimeES
from sktime.transformations.series.outlier_detection import HampelFilter
from statsmodels.tsa.seasonal import STL


IMPUTE_METHODS = {
    "선형 보간 (기본값 권장)": "linear",
    "LOCF — 직전값으로 채움": "ffill",
    "NOCB — 직후값으로 채움": "bfill",
    "이동평균": "mean",
    "지수평활 보간": "forecaster",
}

OUTLIER_METHODS = {
    "Hampel Filter (기본값 권장)": "hampel",
    "STL 잔차 + G-ESD": "gesd",
    "IQR 기반": "iqr",
}

DENOISE_METHODS = {
    "off": "off",
    "단순이동평균 (SMA)": "sma",
    "지수이동평균 (EMA)": "ema",
    "FFT 저주파 통과 (LPF)": "fft_lpf",
    "FFT 고주파 통과 (HPF)": "fft_hpf",
}


# ── 1단계: 결측치 ──────────────────────────────

@st.cache_data(show_spinner=False)
def impute_missing(series: pd.Series, method_label: str, window: int = 3) -> pd.Series:
    method_key = IMPUTE_METHODS[method_label]

    if method_key == "mean":
        filled = series.copy()
        mask = filled.isna()
        rolling = (
            series.ffill().bfill()
            .rolling(window=window, min_periods=1, center=True)
            .mean()
        )
        filled[mask] = rolling[mask]
        return filled

    if method_key == "forecaster":
        imputer = Imputer(method="forecaster", forecaster=SktimeES())
    else:
        imputer = Imputer(method=method_key)

    return imputer.fit_transform(series)


def recommend_impute_method(series: pd.Series, sp: int) -> str:
    if series.isna().sum() == 0:
        return "선형 보간 (기본값 권장)"
    max_consec, cur = 0, 0
    for v in series.isna():
        cur = cur + 1 if v else 0
        max_consec = max(max_consec, cur)
    return (
        "지수평활 보간"
        if max_consec > max(sp // 4, 2)
        else "선형 보간 (기본값 권장)"
    )


# ── 2단계: 이상치 ──────────────────────────────

@st.cache_data(show_spinner=False)
def detect_and_replace_outliers(
    series: pd.Series,
    method_label: str,
    window_length: int = 5,
    n_sigma: float = 3.0,
    sp: int = 12,
) -> tuple[pd.Series, pd.Series]:
    method = OUTLIER_METHODS[method_label]

    if method == "hampel":
        cleaned = HampelFilter(window_length=window_length, n_sigma=n_sigma).fit_transform(series)
        outlier_mask = series.notna() & cleaned.isna()
        cleaned = cleaned.interpolate(method="linear", limit_direction="both")

    elif method == "gesd":
        try:
            import scikit_posthocs as sp_mod

            # 1) STL 분해: 추세 + 계절성 + 잔차
            s = series.dropna()
            stl_result = STL(s, period=sp, robust=True).fit()
            trend    = pd.Series(stl_result.trend,    index=s.index)
            seasonal = pd.Series(stl_result.seasonal, index=s.index)
            resid    = pd.Series(stl_result.resid,    index=s.index)

            # 2) 잔차에서만 G-ESD로 이상치 탐지
            inliers = sp_mod.outliers_gesd(resid, outliers=10, report=False)
            outlier_mask_s = (resid < inliers.min()) | (resid > inliers.max())

            # 3) 이상치 위치의 잔차만 NaN → 선형 보간으로 복구
            resid_fixed = resid.copy()
            resid_fixed[outlier_mask_s] = np.nan
            resid_fixed = resid_fixed.interpolate(method="linear", limit_direction="both")

            # 4) 추세 + 계절성 + 보정된 잔차 재합산으로 원본 스케일 복원
            cleaned_s = trend + seasonal + resid_fixed

            # 원본 series 인덱스에 맞게 재정렬 (dropna로 줄었을 수 있으므로)
            cleaned = series.copy()
            cleaned[cleaned_s.index] = cleaned_s.values
            cleaned = cleaned.interpolate(method="linear", limit_direction="both")

            outlier_mask = pd.Series(False, index=series.index)
            outlier_mask[outlier_mask_s[outlier_mask_s].index] = True

        except ImportError:
            st.warning("scikit-posthocs 미설치. Hampel Filter로 대체합니다.")
            return detect_and_replace_outliers(
                series, "Hampel Filter (기본값 권장)", window_length, n_sigma, sp
            )

    elif method == "iqr":
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        outlier_mask = (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)
        cleaned = series.copy()
        cleaned[outlier_mask] = np.nan
        cleaned = cleaned.interpolate(method="linear", limit_direction="both")

    else:
        cleaned = series.copy()
        outlier_mask = pd.Series(False, index=series.index)

    if not isinstance(outlier_mask, pd.Series):
        outlier_mask = pd.Series(outlier_mask, index=series.index)

    return cleaned, outlier_mask


# ── 3단계: 디노이징 ────────────────────────────

def _fft_filter(series: pd.Series, cutoff: float, mode: str) -> pd.Series:
    """
    FFT 기반 주파수 필터.

    Parameters
    ----------
    cutoff : float
        0.0 ~ 1.0 사이 값. 전체 주파수 성분 중 유지할 비율.
        LPF → 하위 cutoff 비율의 저주파 성분만 통과
        HPF → 상위 (1 - cutoff) 비율의 고주파 성분만 통과
    mode : str
        "lpf" 또는 "hpf"
    """
    values = series.values.astype(float)
    n = len(values)

    # FFT → 주파수 도메인
    freq_domain = np.fft.fft(values)

    # cutoff를 인덱스로 변환
    cutoff_idx = max(1, int(n * cutoff / 2))  # 양측 스펙트럼 기준

    # 마스크 생성 (통과할 주파수 성분만 1)
    mask = np.zeros(n, dtype=bool)
    if mode == "lpf":
        # 저주파 통과: 양 끝(0 ~ cutoff_idx) 유지
        mask[:cutoff_idx] = True
        mask[n - cutoff_idx:] = True
    elif mode == "hpf":
        # 고주파 통과: 가운데 제거, 나머지 유지
        mask[:] = True
        mask[:cutoff_idx] = False
        mask[n - cutoff_idx:] = False

    # 마스크 적용 후 IFFT → 시간 도메인 복원
    filtered = np.fft.ifft(freq_domain * mask).real

    return pd.Series(filtered, index=series.index, name=series.name)


@st.cache_data(show_spinner=False)
def denoise(
    series: pd.Series,
    method_label: str,
    window: int = 5,
    alpha: float = 0.3,
    cutoff: float = 0.1,
) -> pd.Series:
    """
    디노이징. 기본값 off.

    Parameters
    ----------
    window : int
        SMA rolling window 크기
    alpha : float
        EMA 평활 계수 (0~1, 클수록 최근 값 반영 강함)
    cutoff : float
        FFT 필터 차단 주파수 비율 (0~1)
        LPF: 하위 cutoff 비율 저주파만 통과 (작을수록 더 많이 평활)
        HPF: 상위 (1-cutoff) 비율 고주파만 통과 (작을수록 더 많이 평활)
    """
    method = DENOISE_METHODS.get(method_label, "off")

    if method == "off":
        return series

    if method == "sma":
        v = series.rolling(window=window, min_periods=1, center=True).mean()
        return pd.Series(v.values, index=series.index, name=series.name)

    if method == "ema":
        v = series.ewm(alpha=alpha, adjust=True).mean()
        return pd.Series(v.values, index=series.index, name=series.name)

    if method == "fft_lpf":
        return _fft_filter(series, cutoff=cutoff, mode="lpf")

    if method == "fft_hpf":
        return _fft_filter(series, cutoff=cutoff, mode="hpf")

    return series


def calc_noise_reduction_rate(original: pd.Series, denoised: pd.Series) -> float:
    orig_var = original.var()
    if orig_var == 0:
        return 0.0
    return max(0.0, (orig_var - denoised.var()) / orig_var * 100)


# ── 4단계: 정상성 진단 (AutoARIMA 선택 시만) ──

@st.cache_data(show_spinner=False)
def run_adf_test(series: pd.Series) -> dict:
    clean = series.dropna()
    if len(clean) < 10:
        return {"error": "데이터가 너무 짧아 검정 불가 (최소 10개 필요)"}
    try:
        stat, pvalue, lags, nobs, cvs, _ = adfuller(clean.values, autolag="AIC")
        return {
            "statistic": round(float(stat), 4),
            "pvalue": round(float(pvalue), 4),
            "is_stationary": pvalue < 0.05,
            "lags_used": int(lags),
            "n_obs": int(nobs),
            "critical_values": {k: round(v, 4) for k, v in cvs.items()},
            "error": None,
        }
    except Exception as e:
        return {"error": str(e)}


def render_stationarity_result(adf_result: dict):
    """AutoARIMA 선택 시 예측 탭 상단에 인라인 렌더링."""
    if adf_result.get("error"):
        st.warning(f"정상성 검정 오류: {adf_result['error']}")
        return

    is_stat = adf_result["is_stationary"]
    pvalue = adf_result["pvalue"]

    if is_stat:
        st.success(
            f"정상 시계열입니다 (p-value = {pvalue}). "
            "AutoARIMA가 최적 파라미터를 탐색합니다."
        )
        return

    st.warning(f"비정상 시계열 감지 (p-value = {pvalue})")

    with st.expander("정상성 검정 결과 — 자세히 보기", expanded=True):
        c1, c2 = st.columns(2)
        c1.metric("ADF 통계량", adf_result["statistic"])
        c1.metric("p-value", pvalue)
        c2.metric("사용 래그 수", adf_result["lags_used"])
        c2.metric("관측치 수", adf_result["n_obs"])

        st.divider()

        st.markdown("**이게 뭔가요?**")
        st.info(
            "이 데이터는 시간에 따라 평균이나 분산이 변하고 있습니다. "
            "꾸준히 증가하는 추세가 있거나, 특정 시점 이후 변동폭이 커지는 경우가 대표적입니다."
        )

        st.markdown("**문제가 되나요?**")
        st.info(
            "ARIMA는 원래 안정적인 데이터를 가정합니다. "
            "비정상 데이터를 그대로 넣으면 잘못된 파라미터를 추정할 수 있습니다."
        )

        st.markdown("**어떻게 하나요?**")
        st.success(
            "AutoARIMA가 차분(differencing) 횟수 d를 AIC 기준으로 자동으로 찾아 해결합니다. "
            "별도 조치 없이 진행하세요."
        )

        cv = adf_result["critical_values"]
        st.caption(
            f"임계값 — 1%: {cv.get('1%', '-')} | 5%: {cv.get('5%', '-')} | 10%: {cv.get('10%', '-')}"
        )


# ── 통합 실행 ──────────────────────────────────

def run_preprocessing(
    series: pd.Series,
    sp: int,
    impute_method: str,
    impute_window: int,
    outlier_method: str,
    outlier_window: int,
    outlier_sigma: float,
    denoise_method: str,
    denoise_window: int,
    denoise_alpha: float,
    denoise_cutoff: float = 0.1,
) -> dict:
    """결측치 → 이상치 → 디노이징 순서로 처리. 각 단계 결과 반환."""
    imputed = impute_missing(series, impute_method, impute_window)
    cleaned, outlier_mask = detect_and_replace_outliers(
        imputed, outlier_method, outlier_window, outlier_sigma, sp
    )
    final = denoise(cleaned, denoise_method, denoise_window, denoise_alpha, denoise_cutoff)
    noise_rate = (
        calc_noise_reduction_rate(cleaned, final) if denoise_method != "off" else 0.0
    )
    return {
        "original": series,
        "imputed": imputed,
        "cleaned": cleaned,
        "outlier_mask": outlier_mask,
        "final": final,
        "n_outliers": int(outlier_mask.sum()) if isinstance(outlier_mask, pd.Series) else 0,
        "noise_reduction_rate": round(noise_rate, 1),
        "denoise_applied": denoise_method != "off",
    }