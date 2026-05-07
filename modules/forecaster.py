import pandas as pd
import numpy as np
import streamlit as st

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.trend import STLForecaster, PolynomialTrendForecaster
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.transformations.series.detrend import Deseasonalizer, Detrender
from sktime.forecasting.model_selection import temporal_train_test_split


MODEL_REGISTRY = {
    "Naive (기준모델)":           "naive",
    "Holt-Winter's 승법 (권장)":  "holt_winters_mul",
    "Holt-Winter's 가법":         "holt_winters_add",
    "Holt's (선형추세)":           "holts",
    "단순 지수평활 (SES)":         "ses",
    "STL Forecaster (권장)":      "stl",
    "분해법 (Decomp)":             "decomp",
    "AutoARIMA":                   "autoarima",
}

NEEDS_STATIONARITY_CHECK = {"AutoARIMA"}

# 분해법 파라미터 선택지
DECOMP_SEASONAL_MODELS = ["multiplicative", "additive"]
DECOMP_POLY_DEGREES    = [1, 2, 3]


def build_model(
    model_label: str,
    sp: int,
    decomp_seasonal: str = "multiplicative",
    decomp_degree: int = 1,
    series: pd.Series = None,
):
    """
    모델 레이블로 sktime forecaster 인스턴스 반환.

    Parameters
    ----------
    decomp_seasonal : str
        분해법 계절성 모델. "multiplicative" 또는 "additive"
    decomp_degree : int
        분해법 추세 다항식 차수. 1(선형), 2(이차), 3(삼차)
    series : pd.Series, optional
        데이터. 승법 모델 적용 가능 여부 판단에 사용.
    """
    key = MODEL_REGISTRY[model_label]

    # 승법 모델 적용 가능 여부: 0 이하 값이 없어야 함
    def _can_multiplicative(s):
        if s is None:
            return True
        return bool((s.dropna() > 0).all())

    if key == "naive":
        return NaiveForecaster(strategy="mean", window_length=sp)
    elif key == "ses":
        return ExponentialSmoothing(trend=None, seasonal=None, smoothing_level=0.3)
    elif key == "holts":
        return ExponentialSmoothing(trend="add", seasonal=None,
                                    smoothing_level=0.3, smoothing_trend=0.05)
    elif key == "holt_winters_add":
        return ExponentialSmoothing(trend="add", seasonal="add", sp=sp,
                                    smoothing_level=0.3, smoothing_trend=0.05,
                                    smoothing_seasonal=0.05)
    elif key == "holt_winters_mul":
        if _can_multiplicative(series):
            return ExponentialSmoothing(trend="add", seasonal="mul", sp=sp,
                                        smoothing_level=0.3, smoothing_trend=0.05,
                                        smoothing_seasonal=0.05)
        else:
            return ExponentialSmoothing(trend="add", seasonal="add", sp=sp,
                                        smoothing_level=0.3, smoothing_trend=0.05,
                                        smoothing_seasonal=0.05)
    elif key == "stl":
        return STLForecaster(sp=sp)
    elif key == "decomp":
        # 음수/0 포함 시 승법 계절성 불가 → 가법으로 자동 fallback
        safe_seasonal = decomp_seasonal if _can_multiplicative(series) else "additive"
        return TransformedTargetForecaster(steps=[
            ("deseasonalizer", Deseasonalizer(sp=sp, model=safe_seasonal)),
            ("detrender",      Detrender(forecaster=PolynomialTrendForecaster(degree=decomp_degree))),
            ("forecaster",     ExponentialSmoothing()),
        ])
    elif key == "autoarima":
        return AutoARIMA(sp=sp, information_criterion="aic", stepwise=True,
                         suppress_warnings=True, error_action="ignore")
    else:
        raise ValueError(f"알 수 없는 모델: {model_label}")


def split_data(series: pd.Series, horizon: int) -> tuple[pd.Series, pd.Series]:
    """temporal_train_test_split으로 시간 순서 보장 분할."""
    return temporal_train_test_split(series, test_size=horizon)


def make_fh(horizon: int, is_relative: bool = True) -> ForecastingHorizon:
    return ForecastingHorizon(list(range(1, horizon + 1)), is_relative=is_relative)


@st.cache_data(show_spinner=False)
def run_forecast(
    series: pd.Series,
    model_label: str,
    sp: int,
    horizon: int,
    coverage: float = 0.9,
    decomp_seasonal: str = "multiplicative",
    decomp_degree: int = 1,
) -> dict:
    """
    단일 모델 학습·예측.
    반환: y_train, y_test, y_pred, y_interval, y_future, model_info, error
    """
    y_train, y_test = split_data(series, horizon)
    fh = make_fh(len(y_test))
    model = build_model(model_label, sp, decomp_seasonal, decomp_degree, series=y_train)

    try:
        model.fit(y_train)
        y_pred = model.predict(fh)

        # 신뢰구간 (지원 모델만)
        y_interval = None
        try:
            y_interval = model.predict_interval(fh, coverage=coverage)
        except Exception:
            pass

        # 미래 예측: 전체 series로 재학습 후 horizon 스텝 예측        # 수정
        y_future = None
        try:
            model_future = build_model(model_label, sp, decomp_seasonal, decomp_degree, series=series)
            model_future.fit(series)
            fh_future = make_fh(horizon)
            y_future = model_future.predict(fh_future)
        except Exception:
            pass

        # AutoARIMA 파라미터 정보 추출
        model_info = {}
        if MODEL_REGISTRY[model_label] == "autoarima":
            try:
                fp = model.get_fitted_params()
                model_info["order"]          = fp.get("order", "?")
                model_info["seasonal_order"] = fp.get("seasonal_order", "?")
                model_info["aic"]            = round(float(fp["aic"]), 2) if "aic" in fp else "N/A"
                model_info["bic"]            = round(float(fp["bic"]), 2) if "bic" in fp else "N/A"
                model_info["aicc"]           = round(float(fp["aicc"]), 2) if "aicc" in fp else "N/A"
            except Exception:
                pass

        return {
            "y_train":    y_train,
            "y_test":     y_test,
            "y_pred":     y_pred,
            "y_interval": y_interval,
            "y_future":   y_future,
            "model_info": model_info,
            "error":      None,
        }

    except Exception as e:
        return {
            "y_train":    y_train,
            "y_test":     y_test,
            "y_pred":     None,
            "y_interval": None,
            "y_future":   None,
            "model_info": {},
            "error":      str(e),
        }


@st.cache_data(show_spinner=False)
def run_multi_forecast(
    series: pd.Series,
    model_labels: list[str],
    sp: int,
    horizon: int,
    decomp_seasonal: str = "multiplicative",
    decomp_degree: int = 1,
) -> dict[str, dict]:
    """복수 모델 동시 학습·예측. 모델 비교 탭에서 사용."""
    results = {}
    for label in model_labels:
        results[label] = run_forecast(
            series, label, sp, horizon,
            decomp_seasonal=decomp_seasonal,
            decomp_degree=decomp_degree,
        )
    return results