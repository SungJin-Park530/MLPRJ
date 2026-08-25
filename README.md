# IoT Occupancy Prediction

이 프로젝트는 IoT 센서 데이터를 기반으로 실내 재실 여부를 예측하는 머신러닝 애플리케이션입니다.

## 프로젝트 구성

- `iot_streamlit.py`: Streamlit 기반 웹 대시보드
- `dataset/Occupancy_Estimation.csv`: 학습 및 평가에 사용되는 데이터셋
- `model/best_rf_model.joblib`: 학습된 랜덤 포레스트 모델
- `model/features_meta.joblib`: 모델 입력 피처 메타데이터
- `report/eda_report_original.html`: EDA 리포트
- `iot_ml.ipynb`: 실험/분석용 Jupyter 노트북

## 주요 기능

- 센서 값을 입력받아 실내 재실 여부를 예측
- 모델 성능 시각화(혼동 행렬, ROC 곡선)
- Streamlit 기반 인터랙티브 대시보드 제공

## 실행 방법

1. 가상환경 생성

```bash
python -m venv .venv
```

2. 가상환경 활성화

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

3. 패키지 설치

```bash
pip install -r requirements.txt
```

4. 애플리케이션 실행

```bash
streamlit run iot_streamlit.py
```

## 의존성

- Python 3.10 이상 권장
- `requirements.txt` 참고

## 참고

- 본 프로젝트는 재실 여부를 이진 분류 문제로 처리합니다.
- 모델은 `Room_Occupancy_Count` 값을 기반으로 Occupancy를 예측하도록 설계되었습니다.
