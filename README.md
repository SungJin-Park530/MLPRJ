# IoT Occupancy Prediction

IoT 환경 센서 데이터(온도, 조도, CO2, 소리, PIR 모션 감지 등)를 기반으로 실내 재실 여부(Occupied / Not Occupied)를 예측하는 머신러닝 기반 웹 애플리케이션입니다.

## 주요 기능

- 다중 센서 값을 입력받아 실시간으로 재실 여부와 확률(%) 예측
- 랜덤 센서 값 자동 생성 버튼으로 빠른 테스트 지원
- 학습된 모델의 성능을 혼동 행렬(Confusion Matrix), ROC 곡선(AUC)으로 시각화
- Streamlit 기반 대시보드로 별도 프론트엔드 구현 없이 웹 UI 제공

## 서비스 화면

> 스크린샷 추가 예정

| 실시간 예측 화면 | 모델 평가 시각화 화면 |
| --- | --- |
| _(이미지 추가 예정)_ | _(이미지 추가 예정)_ |

## 시연 영상

> 시연 영상(GIF 또는 링크) 추가 예정

## 기술 스택

| 구분 | 기술 |
| --- | --- |
| 언어 | Python 3.10+ |
| 데이터 분석 | pandas, numpy |
| 시각화 | matplotlib, seaborn |
| 머신러닝 | scikit-learn (RandomForestClassifier, GridSearchCV) |
| 모델 직렬화 | joblib |
| 웹 대시보드 | Streamlit |
| 실험/분석 환경 | Jupyter Notebook |

## 아키텍처

```mermaid
flowchart LR
    A[원본 데이터셋<br/>Occupancy_Estimation.csv] --> B[EDA & 전처리<br/>iot_ml.ipynb]
    B --> C[모델 학습 & 튜닝<br/>RandomForest + GridSearchCV]
    C --> D[모델 직렬화<br/>best_rf_model.joblib<br/>features_meta.joblib]
    D --> E[Streamlit 앱<br/>iot_streamlit.py]
    E --> F[사용자 브라우저]
```

- **분석 단계**: `iot_ml.ipynb`에서 결측치/중복/다중공선성(VIF) 점검, 상관관계 분석, 모델 비교 및 하이퍼파라미터 튜닝을 수행합니다.
- **서빙 단계**: 학습이 끝난 모델과 피처 메타데이터를 `model/`에 저장하고, `iot_streamlit.py`가 이를 로드하여 실시간 추론 및 시각화를 제공합니다.

## 프로젝트 구성

```
├── iot_streamlit.py                # Streamlit 웹 대시보드
├── iot_ml.ipynb                    # EDA 및 모델 학습 노트북
├── requirements.txt                # 의존성 목록
├── dataset/
│   └── Occupancy_Estimation.csv    # 학습/평가용 데이터셋 (전처리됨)
├── model/
│   ├── best_rf_model.joblib        # 학습된 RandomForest 모델
│   └── features_meta.joblib        # 모델 입력 피처 메타데이터
└── report/
    └── eda_report_original.html    # EDA 리포트
```

## 실행 방법

### 1. 사전 요구사항

- Python 3.10 이상
- pip

### 2. 저장소 클론

```bash
git clone <repository-url>
cd MLPRJ
```

### 3. 가상환경 생성

```bash
python -m venv .venv
```

### 4. 가상환경 활성화

OS 및 셸 환경에 따라 명령어가 다릅니다.

**Windows (PowerShell)**

```powershell
.\.venv\Scripts\Activate.ps1
```

> PowerShell 스크립트 실행이 차단된 경우(보안 정책 오류), 아래 명령어로 현재 사용자에 한해 실행 정책을 완화한 뒤 다시 시도하세요.
>
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

**Windows (명령 프롬프트, cmd.exe)**

```cmd
.venv\Scripts\activate.bat
```

**Windows (Git Bash)**

```bash
source .venv/Scripts/activate
```

**macOS / Linux (bash, zsh)**

```bash
source .venv/bin/activate
```

### 5. 패키지 설치

```bash
pip install -r requirements.txt
```

### 6. 애플리케이션 실행

```bash
streamlit run iot_streamlit.py
```

실행 후 브라우저에서 `http://localhost:8501` 로 자동 접속됩니다.

## 참고 및 라이선스

- 본 프로젝트는 **비영리 목적**의 학습/포트폴리오용 프로젝트입니다.
- 원본 데이터셋은 [UCI Machine Learning Repository - Room Occupancy Estimation Data Set](https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation)을 기반으로 하며, **CC BY 4.0(Creative Commons Attribution 4.0 International)** 라이선스를 따릅니다.
- 본 프로젝트에서 사용한 `dataset/Occupancy_Estimation.csv`는 원본 데이터를 그대로 사용한 것이 아니라, 머신러닝 학습에 적합하도록 **일부 전처리 및 변형**(타겟 컬럼 이진화 등)을 거친 데이터입니다.
- 원본 데이터 출처 및 라이선스 조건(저작자 표시 등)은 CC BY 4.0 원칙을 따릅니다.
