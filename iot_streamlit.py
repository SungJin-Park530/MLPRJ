import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# ---------------------------------------------------------
# 1. Page Configuration & Caching Functions
# ---------------------------------------------------------
st.set_page_config(page_title="IoT Occupancy Dashboard", layout="wide")

@st.cache_resource
def load_model_and_meta():
    """모델 및 피처 메타데이터 로드 (resource 캐싱)"""
    model = joblib.load('model/best_rf_model.joblib')
    meta = joblib.load('model/features_meta.joblib')
    return model, meta

@st.cache_data
def load_sample_data():
    df = pd.read_csv('dataset/Occupancy_Estimation.csv') 
    
    # 1) 원본 타겟 컬럼(Room_Occupancy_Count)을 이진 분류용(Occupancy)으로 변환
    if 'Room_Occupancy_Count' in df.columns:
        df['Occupancy'] = (df['Room_Occupancy_Count'] > 0).astype(int)
    
    # 2) 학습에 사용하지 않은 불필요한 컬럼 제거
    drop_cols = ['Date', 'Time', 'Room_Occupancy_Count', 'Occupancy']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    
    # 3) 종속변수 추출
    y = df['Occupancy']
    
    return X, y

# 객체 로드
try:
    model, meta = load_model_and_meta()
    feature_names = meta['feature_names']
except Exception as e:
    st.error(f"모델 로드 실패: {e}. 'model/' 경로의 joblib 파일을 확인하세요.")
    st.stop()

# ---------------------------------------------------------
# 2. Layout (Simple 2-Tab Structure)
# ---------------------------------------------------------
st.title("IoT 기반 실내 재실 여부 예측 시스템")

tab1, tab2 = st.tabs(["🔮 실시간 재실 예측", "📊 모델 평가 시각화"])

# ---------------------------------------------------------
# Tab 1: 실시간 재실 예측 (Real-time Prediction)
# ---------------------------------------------------------
# ---------------------------------------------------------
# Tab 1: 실시간 재실 예측 (Real-time Prediction)
# ---------------------------------------------------------
with tab1:
    st.subheader("센서 데이터 입력")
    
    # 평가용 데이터셋을 참고하여 최소/최대 범위 파악
    X_sample, _ = load_sample_data()
    
    # 1. 랜덤 데이터 생성 버튼
    if st.button("🎲 랜덤 센서 값 불러오기"):
        for feature in feature_names:
            min_val = float(X_sample[feature].min())
            max_val = float(X_sample[feature].max())
            # 세션 상태에 무작위 값 저장
            st.session_state[f"input_{feature}"] = round(np.random.uniform(min_val, max_val), 2)
            
    # 2. 동적 입력 필드 생성
    input_data = {}
    col1, col2, col3 = st.columns(3)
    
    for idx, feature in enumerate(feature_names):
        target_col = col1 if idx % 3 == 0 else col2 if idx % 3 == 1 else col3
        
        # 세션에 저장된 값이 없으면 데이터셋의 평균값을 기본값으로 세팅
        default_val = st.session_state.get(
            f"input_{feature}", 
            round(float(X_sample[feature].mean()), 2)
        )
        
        input_data[feature] = target_col.number_input(
            f"{feature}", 
            value=default_val,
            key=f"input_{feature}", # session_state 연동용 key
            step=0.1
        )
    
    # DataFrame 변환 및 컬럼 순서 맞춤
    input_df = pd.DataFrame([input_data])[feature_names]
    
    st.divider()
    
    if st.button("재실 여부 예측 실행", type="primary"):
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            if prediction == 1:
                st.error("🚨 **현재 상태: 재실 (Occupied)**")
            else:
                st.success("🍃 **현재 상태: 공실 (Not Occupied)**")
                
        with col_res2:
            st.metric(label="재실 확률 (Occupancy Probability)", value=f"{proba * 100:.2f} %")

# ---------------------------------------------------------
# Tab 2: 모델 평가 및 성능 분석 (Evaluation)
# ---------------------------------------------------------
with tab2:
    st.subheader("모델 성능 검증 시각화")
    
    try:
        X_test, y_test = load_sample_data()
        X_test = X_test[feature_names] # 학습 피처 구조 동기화
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        col_graph1, col_graph2 = st.columns(2)
        
        # 1) Confusion Matrix
        with col_graph1:
            st.markdown("**1. 혼동 행렬 (Confusion Matrix)**")
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm,
                xticklabels=['Not Occupied', 'Occupied'],
                yticklabels=['Not Occupied', 'Occupied']
            )
            ax_cm.set_xlabel('Predicted')
            ax_cm.set_ylabel('True')
            st.pyplot(fig_cm)
            
        # 2) ROC Curve
        with col_graph2:
            st.markdown("**2. ROC 곡선 (ROC Curve)**")
            fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_val = roc_auc_score(y_test, y_proba)
            
            ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc_val:.4f}')
            ax_roc.plot([0, 1], [0, 1], color='navy', linestyle='--')
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.legend(loc="lower right")
            ax_roc.grid(True, linestyle="--", alpha=0.3)
            st.pyplot(fig_roc)

    except FileNotFoundError:
        st.warning("`iot_occupancy_data.csv` 데이터 파일을 찾을 수 없어 평가 시각화를 생략합니다.")