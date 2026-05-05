import pandas as pd
import numpy as np
import streamlit as st

from sktime.performance_metrics.forecasting import (
    mean_absolute_error as MAE,
    mean_squared_error as MSE,
    mean_absolute_percentage_error as MAPE,
    median_absolute_error as MdAE,
    mean_absolute_scaled_error as MASE,
)


def calc_metrics(
    y_test: pd.Series,
    y_pred: pd.Series,
    y_train: pd.Series,
) -> dict:
    """강의록 평가지표 전부 계산."""
    if y_pred is None:
        return {}
    try:
        mae   = round(float(MAE(y_test, y_pred)), 4)
        rmse  = round(float(np.sqrt(MSE(y_test, y_pred))), 4)
        mape  = round(float(MAPE(y_test, y_pred)) * 100, 4)
        smape = round(float(MAPE(y_test, y_pred, symmetric=True)) * 100, 4)
        mdae  = round(float(MdAE(y_test, y_pred)), 4)
        mase  = round(float(MASE(y_test, y_pred, y_train=y_train)), 4)

        # 편향 지표
        residuals = y_test - y_pred
        rsfe  = round(float(residuals.sum()), 4)
        ts_val = round(float(rsfe / mae) if mae != 0 else 0.0, 4)

        return {
            "MAE": mae,
            "RMSE": rmse,
            "MAPE(%)": mape,
            "SMAPE(%)": smape,
            "MdAE": mdae,
            "MASE": mase,
            "RSFE": rsfe,
            "TS": ts_val,
        }
    except Exception as e:
        return {"error": str(e)}


def build_comparison_table(results: dict[str, dict], y_train: pd.Series) -> pd.DataFrame:
    """복수 모델 평가지표 비교 DataFrame 생성."""
    rows = []
    for label, res in results.items():
        if res.get("error") or res.get("y_pred") is None:
            rows.append({"모델": label, "MAE": None, "RMSE": None,
                         "MAPE(%)": None, "MASE": None, "오류": res.get("error", "예측 실패")})
            continue
        m = calc_metrics(res["y_test"], res["y_pred"], y_train)
        if "error" in m:
            rows.append({"모델": label, "오류": m["error"]})
        else:
            rows.append({"모델": label, **m})
    df = pd.DataFrame(rows).set_index("모델")
    return df


def get_best_model(comparison_df: pd.DataFrame, metric: str = "MAPE(%)") -> str | None:
    """지정 지표 기준 최적 모델 반환."""
    if metric not in comparison_df.columns:
        return None
    valid = comparison_df[metric].dropna()
    if len(valid) == 0:
        return None
    return str(valid.idxmin())


def render_metrics_cards(metrics: dict):
    """평가지표를 Streamlit metric 카드로 렌더링."""
    if not metrics or "error" in metrics:
        st.error(f"평가지표 계산 실패: {metrics.get('error', '')}")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", metrics.get("MAE", "-"))
    c2.metric("RMSE", metrics.get("RMSE", "-"))
    c3.metric("MAPE", f"{metrics.get('MAPE(%)', '-')}%")
    c4.metric("SMAPE", f"{metrics.get('SMAPE(%)', '-')}%")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("MdAE", metrics.get("MdAE", "-"))
    c6.metric("MASE", metrics.get("MASE", "-"))
    c7.metric("RSFE", metrics.get("RSFE", "-"))

    ts = metrics.get("TS", 0)

    c8.metric("Tracking Signal", ts, help="|TS| > 4이면 편향 의심")

    if abs(ts) > 4:
        st.warning(
            f"Tracking Signal = {ts} → |TS| > 4로 편향이 의심됩니다. "
            "예측이 지속적으로 과대 또는 과소 추정되고 있을 수 있습니다."
        )
