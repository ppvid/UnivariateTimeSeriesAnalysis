import chardet
import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO


FREQ_SP_MAP = {
    "D": 7, "W": 52,
    "M": 12, "ME": 12, "MS": 12,
    "Q": 4, "QE": 4, "QS": 4,
    "Y": 1, "YE": 1, "YS": 1,
    "H": 24, "h": 24,
    "T": 60, "min": 60,
}

FREQ_LABEL_MAP = {
    "D": "일별", "W": "주별",
    "M": "월별", "ME": "월별", "MS": "월별",
    "Q": "분기별", "QE": "분기별", "QS": "분기별",
    "Y": "연별", "YE": "연별", "YS": "연별",
    "H": "시간별", "h": "시간별",
    "T": "분별", "min": "분별",
}

# sktime PeriodIndex 변환에 쓸 정규 freq 매핑
# pandas infer_freq는 "MS","QS","YS" 등 Start 계열을 반환하지만
# PeriodIndex는 "M","Q","Y"만 지원하므로 여기서 변환
FREQ_PERIOD_MAP = {
    "D":  "D",
    "W":  "W",
    "M":  "M",  "ME": "M",  "MS": "M",
    "Q":  "Q",  "QE": "Q",  "QS": "Q",
    "Y":  "Y",  "YE": "Y",  "YS": "Y",
    "A":  "Y",  "AE": "Y",  "AS": "Y",  # alias
    "H":  "H",  "h":  "H",
    "T":  "T",  "min":"T",
}


def detect_encoding(file_bytes: bytes) -> str:
    result = chardet.detect(file_bytes)
    encoding = result.get("encoding", "utf-8") or "utf-8"
    if encoding.lower() in ("ascii", "unknown"):
        encoding = "utf-8"
    return encoding


def try_read_csv(file_bytes: bytes) -> tuple[pd.DataFrame, str]:
    encoding = detect_encoding(file_bytes)
    for enc in list(dict.fromkeys([encoding, "utf-8", "cp949", "euc-kr"])):
        try:
            df = pd.read_csv(BytesIO(file_bytes), encoding=enc, sep=None, engine="python")
            return df, enc
        except Exception:
            continue
    raise ValueError("CSV 파일을 읽을 수 없습니다. 인코딩을 확인해 주세요.")


def detect_date_column(df: pd.DataFrame) -> str | None:
    best_col, best_rate = None, 0.0
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        try:
            converted = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
            rate = converted.notna().mean()
            if rate > 0.8 and rate > best_rate:
                best_rate, best_col = rate, col
        except Exception:
            continue
    return best_col


def detect_value_column(df: pd.DataFrame, date_col: str | None) -> str | None:
    for col in df.columns:
        if col == date_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].isna().mean() < 0.5:
            return col
    return None


def get_candidate_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    date_cands, value_cands = [], []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            value_cands.append(col)
        else:
            try:
                rate = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce").notna().mean()
                if rate > 0.5:
                    date_cands.append(col)
            except Exception:
                pass
    return date_cands, value_cands


def infer_freq_from_index(idx: pd.DatetimeIndex) -> str | None:
    """pd.infer_freq 실패 시 날짜 간격 최빈값으로 추정."""
    freq = pd.infer_freq(idx)
    if freq:
        return freq
    if len(idx) < 2:
        return None
    diffs = idx[1:] - idx[:-1]
    mode = pd.Series(diffs).mode()
    if len(mode) == 0:
        return None
    days = mode.iloc[0].days
    if days == 1:      return "D"
    elif days <= 8:    return "W"
    elif days <= 31:   return "MS"
    elif days <= 92:   return "QS"
    elif days <= 366:  return "YS"
    return None


def get_sp(freq: str | None) -> int:
    if not freq:
        return 1
    return FREQ_SP_MAP.get(freq.rstrip("0123456789-"), 1)


def get_freq_label(freq: str | None) -> str:
    if not freq:
        return "불규칙"
    return FREQ_LABEL_MAP.get(freq.rstrip("0123456789-"), freq)


def ensure_freq(series: pd.Series, freq: str | None) -> pd.Series:
    """
    series 인덱스에 freq 정보를 보장.
    PeriodIndex → 그대로
    DatetimeIndex with freq → 그대로
    DatetimeIndex without freq → PeriodIndex로 변환
    """
    if isinstance(series.index, pd.PeriodIndex):
        return series

    if isinstance(series.index, pd.DatetimeIndex):
        if series.index.freq is not None:
            return series
        if freq is None:
            return series
        # FREQ_PERIOD_MAP에서 직접 조회 (rstrip 방식 대신)
        period_freq = FREQ_PERIOD_MAP.get(freq) or FREQ_PERIOD_MAP.get(freq.rstrip("0123456789-"), "D")
        try:
            period_index = series.index.to_period(period_freq)
            return pd.Series(series.values, index=period_index, name=series.name)
        except Exception:
            try:
                return series.asfreq(freq)
            except Exception:
                return series

    return series


@st.cache_data(show_spinner=False)
def load_and_parse(file_bytes: bytes, date_col: str, value_col: str) -> dict:
    df, _ = try_read_csv(file_bytes)

    df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True, errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.set_index(date_col)

    dt_index = pd.DatetimeIndex(df.index)
    freq = infer_freq_from_index(dt_index)
    sp = get_sp(freq)
    freq_label = get_freq_label(freq)

    # PeriodIndex 변환 시도 (sktime 호환)
    period_freq = FREQ_PERIOD_MAP.get(freq) or FREQ_PERIOD_MAP.get(freq.rstrip("0123456789-")) if freq else None
    series = None

    if period_freq:
        try:
            period_index = dt_index.to_period(period_freq)
            series = pd.Series(df[value_col].values, index=period_index, name=value_col)
        except Exception:
            pass

    if series is None:
        # PeriodIndex 실패 → DatetimeIndex에 freq 강제 설정
        try:
            reindexed = pd.date_range(
                start=dt_index[0], periods=len(dt_index), freq=freq or "D"
            )
            series = pd.Series(df[value_col].values, index=reindexed.to_period(period_freq or "D"), name=value_col)
        except Exception:
            # 최후 수단: DatetimeIndex 그대로
            series = pd.Series(df[value_col].values, index=dt_index, name=value_col)

    n_total = len(series)
    n_missing = int(series.isna().sum())

    return {
        "series":       series,
        "freq":         freq,
        "sp":           sp,
        "freq_label":   freq_label,
        "n_total":      n_total,
        "n_missing":    n_missing,
        "missing_rate": n_missing / n_total if n_total > 0 else 0.0,
        "has_duplicate": bool(series.index.duplicated().any()),
        "has_negative":  bool((series.dropna() < 0).any()),
    }