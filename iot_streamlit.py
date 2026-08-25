import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import math

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# ---------------------------------------------------------
# 1. Page Configuration & Caching Functions
# ---------------------------------------------------------
st.set_page_config(page_title="IoT Occupancy Dashboard", layout="wide")

# 전체 페이지 콘텐츠 영역의 최대 너비를 1000px로 제한
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1000px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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

tab1, tab2 = st.tabs(["🔮 실시간 재실 예측", "📊 데이터 인사이트 대시보드"])

# ---------------------------------------------------------
# Tab 1: 실시간 재실 예측 (Real-time Prediction)
# ---------------------------------------------------------
# ---------------------------------------------------------
# Tab 1: 실시간 재실 예측 (Real-time Prediction)
# ---------------------------------------------------------
with tab1:
    st.subheader("센서 데이터 입력")
    
    # 예측 결과가 표시될 영역을 미리 확보 (버튼 클릭 후 이 자리에 결과 렌더링)
    result_placeholder = st.empty()
    with result_placeholder.container():
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.info("⏳ **예측을 실행하면 여기에 결과가 표시됩니다.**")
        with col_res2:
            st.metric(label="재실 확률 (Occupancy Probability)", value="?%")
        st.divider()
    
    # 평가용 데이터셋을 참고하여 최소/최대 범위 파악
    X_sample, _ = load_sample_data()
    
    # 1. 예측 실행 버튼 & 랜덤 데이터 생성 버튼
    col_btn1, col_btn2 = st.columns(2)
    predict_clicked = col_btn1.button("재실 여부 예측 실행", type="primary")
    if col_btn2.button("🎲 랜덤 센서 값 불러오기"):
        for feature in feature_names:
            min_val = float(X_sample[feature].min())
            max_val = float(X_sample[feature].max())
            # 세션 상태에 무작위 값 저장 (슬라이더가 있는 그룹은 슬라이더 상태도 함께 동기화)
            random_val = round(np.random.uniform(min_val, max_val), 2)
            st.session_state[f"input_{feature}"] = random_val
            st.session_state[f"slider_{feature}"] = random_val
            
    # 2. 센서 그룹별 동적 입력 필드 생성 (전체 2열 x 그룹 내부 2열 레이아웃)
    input_data = {}
    sensor_groups = [
        ("Temp", "🌡️ 온도"),
        ("Light", "💡 조도"),
        ("Sound", "🔊 소음"),
        ("CO2", "🌫️ CO2"),
        ("PIR", "🚶 적외선 감지 (PIR)"),
    ]
    # 온도-조도, 소음-CO2가 같은 행에 나란히 배치되고, PIR은 마지막 행에 단독 배치
    group_pairs = [(sensor_groups[0], sensor_groups[1]), (sensor_groups[2], sensor_groups[3]), (sensor_groups[4], None)]
    
    # 화면 표시용 한글 라벨 (실제 데이터/컬럼명은 변경하지 않음)
    korean_group_labels = {
        "Temp": "온도",
        "Light": "조도",
        "Sound": "소음",
        "PIR": "적외선 감지",
    }
    
    # 슬라이더를 제공할 그룹과 범위 (기존 데이터를 모두 포괄하는 상식적인 수준의 범위)
    slider_ranges = {
        "Temp": (15.0, 35.0),   # 실내 온도 상식 범위(°C), 실측 24.44~29.0 포괄
        "Light": (0.0, 500.0),  # 실내 조도 상식 범위(lux), 실측 0~280 포괄
    }
    
    # [낮음/보통/높음] 토글을 제공할 그룹과, 각 단계가 가리키는 최소~최대 구간 내 비율
    level_options = ["낮음", "보통", "높음"]
    level_ratios = {"낮음": 0.25, "보통": 0.50, "높음": 0.75}
    level_toggle_groups = ("Sound", "CO2")
    
    def _sync_slider_to_input(feature):
        st.session_state[f"input_{feature}"] = st.session_state[f"slider_{feature}"]
    
    def _sync_input_to_slider(feature):
        st.session_state[f"slider_{feature}"] = st.session_state[f"input_{feature}"]
    
    def _apply_level(keyword, group_features):
        level = st.session_state[f"level_{keyword}"]
        if level is None:
            return
        ratio = level_ratios[level]
        for feature in group_features:
            # 데이터 개수와 무관하게, 최소~최대 구간을 비율로 나눈 위치의 값을 사용
            min_val = float(X_sample[feature].min())
            max_val = float(X_sample[feature].max())
            base_val = min_val + ratio * (max_val - min_val)
            # 동일한 단계를 선택해도 값이 약간 달라지도록 미세한 무작위 변동 추가
            jittered_val = base_val + np.random.uniform(-0.05, 0.05) * base_val
            st.session_state[f"input_{feature}"] = round(jittered_val, 2)
    
    def group_row_count(keyword):
        return math.ceil(len([f for f in feature_names if keyword in f]) / 2)
    
    def render_sensor_group(keyword, group_title, target_rows):
        group_features = [f for f in feature_names if keyword in f]
        if not group_features:
            return
        
        st.markdown(f"##### {group_title}")
        
        if keyword in level_toggle_groups:
            st.segmented_control(
                "레벨 선택",
                options=level_options,
                key=f"level_{keyword}",
                on_change=_apply_level,
                args=(keyword, tuple(group_features)),
                label_visibility="collapsed",
            )
        
        icol1, icol2 = st.columns(2)
        has_slider = keyword in slider_ranges
        for idx, feature in enumerate(group_features):
            target_col = icol1 if idx % 2 == 0 else icol2
            
            # 세션에 저장된 값이 없으면 데이터셋의 평균값을 기본값으로 세팅
            default_val = st.session_state.get(
                f"input_{feature}", 
                round(float(X_sample[feature].mean()), 2)
            )
            
            if keyword == "CO2":
                display_label = "CO2 변화율" if "Slope" in feature else "CO2"
            else:
                display_label = f"{korean_group_labels[keyword]} {idx + 1}"
            
            target_col.markdown(f"**{display_label}**")
            
            if has_slider:
                slider_min, slider_max = slider_ranges[keyword]
                if f"slider_{feature}" not in st.session_state:
                    st.session_state[f"slider_{feature}"] = min(max(default_val, slider_min), slider_max)
                target_col.slider(
                    display_label,
                    min_value=slider_min,
                    max_value=slider_max,
                    step=0.1,
                    key=f"slider_{feature}",
                    on_change=_sync_slider_to_input,
                    args=(feature,),
                    label_visibility="collapsed",
                )
            
            input_data[feature] = target_col.number_input(
                display_label, 
                value=default_val,
                key=f"input_{feature}", # session_state 연동용 key
                step=0.1,
                label_visibility="collapsed",
                on_change=_sync_input_to_slider if has_slider else None,
                args=(feature,) if has_slider else None,
            )
        
        # 옆 영역보다 필드 수가 적어 낮은 경우, 높이를 맞추기 위한 여백 삽입
        rows = math.ceil(len(group_features) / 2)
        for _ in range(target_rows - rows):
            icol1.markdown("<div style='height:68px'></div>", unsafe_allow_html=True)
            icol2.markdown("<div style='height:68px'></div>", unsafe_allow_html=True)
    
    for pair_idx, (group_a, group_b) in enumerate(group_pairs):
        if pair_idx > 0:
            st.markdown("---")
        
        target_rows = max(
            group_row_count(group_a[0]),
            group_row_count(group_b[0]) if group_b else 0
        )
        
        outer_col1, outer_col2 = st.columns(2)
        with outer_col1:
            render_sensor_group(group_a[0], group_a[1], target_rows)
        if group_b:
            with outer_col2:
                render_sensor_group(group_b[0], group_b[1], target_rows)
    
    # DataFrame 변환 및 컬럼 순서 맞춤
    input_df = pd.DataFrame([input_data])[feature_names]
    
    if predict_clicked:
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]
        
        with result_placeholder.container():
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                if prediction == 1:
                    st.error("🚨 **현재 상태: 재실 (Occupied)**")
                else:
                    st.success("🍃 **현재 상태: 공실 (Not Occupied)**")
                    
            with col_res2:
                st.metric(label="재실 확률 (Occupancy Probability)", value=f"{proba * 100:.2f} %")
            st.divider()
    # predict_clicked가 False인 경우, 위에서 미리 채운 안내 문구가 그대로 유지됨

# ---------------------------------------------------------
# Tab 2: 데이터 인사이트 대시보드 (Model Insight & Evaluation)
# ---------------------------------------------------------
with tab2:
    st.subheader("모델 성능 및 특성 인사이트")
    
    try:
        X_test, y_test = load_sample_data()
        X_test = X_test[feature_names] # 학습 피처 구조 동기화
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        sub_tab1, sub_tab2, sub_tab3 = st.tabs(["1. 혼동 행렬", "2. ROC 곡선", "3. Feature Importance"])
        
        # 이미지 크기 옵션 (선택 비율만큼 원본 figsize에 곱해 그래프 크기를 조절)
        size_scale_map = {"50%": 0.5, "75%": 0.75, "100%": 1.0}
        
        # 1) Confusion Matrix
        with sub_tab1:
            st.caption("모델이 실제 재실/공실 여부를 얼마나 정확히 맞췄는지 보여주는 표입니다.")
            cm_size_option = st.segmented_control("이미지 크기", options=list(size_scale_map.keys()), default="50%", key="cm_size")
            cm_scale = size_scale_map[cm_size_option]
            fig_cm, ax_cm = plt.subplots(figsize=(5 * cm_scale, 4 * cm_scale))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm,
                xticklabels=['Not Occupied', 'Occupied'],
                yticklabels=['Not Occupied', 'Occupied']
            )
            ax_cm.set_xlabel('Predicted')
            ax_cm.set_ylabel('True')
            st.pyplot(fig_cm, use_container_width=False)
            
        # 2) ROC Curve
        with sub_tab2:
            st.caption("분류 임계값 변화에 따른 모델의 판별 성능(AUC)을 나타내는 곡선입니다.")
            roc_size_option = st.segmented_control("이미지 크기", options=list(size_scale_map.keys()), default="50%", key="roc_size")
            roc_scale = size_scale_map[roc_size_option]
            fig_roc, ax_roc = plt.subplots(figsize=(5 * roc_scale, 4 * roc_scale))
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_val = roc_auc_score(y_test, y_proba)
            
            ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc_val:.4f}')
            ax_roc.plot([0, 1], [0, 1], color='navy', linestyle='--')
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.legend(loc="lower right")
            ax_roc.grid(True, linestyle="--", alpha=0.3)
            st.pyplot(fig_roc, use_container_width=False)

        # 3) Feature Importance
        with sub_tab3:
            st.caption("모델의 예측에 각 센서 피처가 얼마나 큰 영향을 미쳤는지 보여줍니다.")
            fi_size_option = st.segmented_control("이미지 크기", options=list(size_scale_map.keys()), default="50%", key="fi_size")
            fi_scale = size_scale_map[fi_size_option]
            importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
            fig_fi, ax_fi = plt.subplots(figsize=(6 * fi_scale, max(4, len(importances) * 0.4) * fi_scale))
            ax_fi.barh(importances.index, importances.values, color='seagreen')
            ax_fi.set_xlabel('Importance')
            st.pyplot(fig_fi, use_container_width=False)

    except FileNotFoundError:
        st.warning("`iot_occupancy_data.csv` 데이터 파일을 찾을 수 없어 평가 시각화를 생략합니다.")