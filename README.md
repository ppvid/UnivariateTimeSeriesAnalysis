# 시계열 예측 대시보드

sktime 기반 단변량 시계열 자동 분석 및 예측 Streamlit 앱.

## 실행 방법

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud 배포

1. GitHub에 이 폴더를 push
2. [share.streamlit.io](https://share.streamlit.io) 에서 New app
3. `app.py` 지정 후 Deploy

## 파일 구조

```
ts_forecast_app/
├── app.py                  # 메인 앱
├── requirements.txt
├── README.md
└── modules/
    ├── data_loader.py      # CSV 파싱 · 컬럼 감지 · 주기 추론
    ├── preprocessor.py     # 결측치 · 이상치 · 디노이징 · 정상성
    ├── forecaster.py       # sktime 모델 학습 · 예측
    └── evaluator.py        # 평가지표 계산 · 시각화
```

## 지원 모델 (sktime 기반)

| 모델 | 클래스 |
|------|--------|
| Naive | `NaiveForecaster` |
| SES / Holt / Holt-Winter's | `ExponentialSmoothing` |
| STL Forecaster | `STLForecaster` |
| 분해법 파이프라인 | `Deseasonalizer + Detrender + ES` |
| AutoARIMA | `AutoARIMA` (pmdarima 래핑) |

## 전처리 흐름

결측치 처리 → 이상치 탐지·대체 → 디노이징(선택) → 정상성 진단(AutoARIMA 시만)
