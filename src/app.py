import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# Import các hàm từ backend (Giữ nguyên file cũ, không đụng tới)
from data_loader import load_data, min_max_scaler
from knn_model import KNN_Classifier

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(page_title="Clinical Architect AI", page_icon="🩺", layout="wide")

# --- XỬ LÝ DỮ LIỆU NGẦM (BACKEND LOGIC) ---
@st.cache_data
def init_system():
    # 1. Tải dữ liệu và Model
    X_raw, y = load_data('data.csv')
    min_vals = np.min(X_raw, axis=0)
    max_vals = np.max(X_raw, axis=0)
    
    # 2. Huấn luyện Model
    X_scaled = min_max_scaler(X_raw)
    model = KNN_Classifier(k=5)
    model.fit(X_scaled, y)
    
    # 3. Tính toán dữ liệu trung bình cho Biểu đồ Radar XAI
    m_avg_scaled = (np.mean(X_raw[y == 1], axis=0) - min_vals) / (max_vals - min_vals + 1e-9)
    b_avg_scaled = (np.mean(X_raw[y == 0], axis=0) - min_vals) / (max_vals - min_vals + 1e-9)
    
    return X_raw, y, X_scaled, min_vals, max_vals, model, m_avg_scaled, b_avg_scaled

# Khởi chạy hệ thống
try:
    X_raw, y_raw, X_scaled_full, min_vals, max_vals, model, m_avg_scaled, b_avg_scaled = init_system()
except Exception as e:
    st.error("Lỗi tải dữ liệu. Hãy chắc chắn file data.csv đang nằm cùng thư mục với app.py")
    st.stop()

# --- GIAO DIỆN CHÍNH (FRONTEND) ---

# ==========================================
# SIDEBAR: KHU VỰC NHẬP LIỆU (CỐ ĐỊNH BÊN TRÁI)
# ==========================================
st.sidebar.markdown("### 🩺 Clinical Architect AI")
st.sidebar.markdown("**v1.0.4 PRECISION ENGINE**")
st.sidebar.markdown("---")

new_patient_data = []
feature_names = ["Radius", "Texture", "Perimeter", "Area", "Smoothness", "Compactness", "Concavity", "Concave Points", "Symmetry", "Fractal Dim"]

# Nhóm 1: Mean
with st.sidebar.expander("📊 MEAN INDICATORS", expanded=True):
    for i in range(10):
        # LỚP GIÁP 1: Ràng buộc Min/Max
        val = st.slider(f"{feature_names[i]} Mean", float(min_vals[i]), float(max_vals[i]), float(np.mean(X_raw[:, i])))
        new_patient_data.append(val)

# Nhóm 2: SE
with st.sidebar.expander("🔬 SE INDICATORS"):
    for i in range(10):
        val = st.slider(f"{feature_names[i]} SE", float(min_vals[i+10]), float(max_vals[i+10]), float(np.mean(X_raw[:, i+10])))
        new_patient_data.append(val)

# Nhóm 3: Worst
with st.sidebar.expander("⚠️ WORST INDICATORS"):
    for i in range(10):
        val = st.slider(f"{feature_names[i]} Worst", float(min_vals[i+20]), float(max_vals[i+20]), float(np.mean(X_raw[:, i+20])))
        new_patient_data.append(val)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.sidebar.button("PREDICT DIAGNOSIS", type="primary", use_container_width=True)


# ==========================================
# MAIN AREA: KHU VỰC HIỂN THỊ KẾT QUẢ
# ==========================================
if predict_btn:
    # 1. Tiền xử lý & Dự đoán
    patient_array = np.array([new_patient_data])
    # LỚP GIÁP 2: Chuẩn hóa
    patient_scaled = (patient_array - min_vals) / (max_vals - min_vals + 1e-9)
    prediction = model.predict(patient_scaled)[0]
    
    # 2. Banner Cảnh báo
    if prediction == 1:
        st.error("### 🚨 MALIGNANT TUMOR DETECTED\nCritical patient profile matches aggressive pathological markers. Immediate clinical intervention recommended.")
        patient_color = '#b81120' # Màu đỏ Primary
    else:
        st.success("### ✅ BENIGN PROFILE\nPatient clinical parameters align with benign historical cases. Routine monitoring advised.")
        patient_color = '#006d41' # Màu xanh Secondary

    st.markdown("<br>", unsafe_allow_html=True)
    
    # 3. Biểu đồ Radar (Y hệt bản vẽ Stitch)
    st.markdown("##### TUMOR MORPHOLOGY PROFILE COMPARISON")
    
    # Chọn 5 chỉ số nổi bật để vẽ Radar: Radius, Texture, Area, Concavity, Smoothness
    categories = ['Radius', 'Texture', 'Area', 'Concavity', 'Smoothness']
    idx = [0, 1, 3, 6, 4]
    
    df_radar = pd.DataFrame({
        'Feature': categories * 3,
        'Value': np.concatenate([patient_scaled[0][idx], m_avg_scaled[idx], b_avg_scaled[idx]]),
        'Group': (['Current Patient'] * 5) + (['Malignant Avg.'] * 5) + (['Benign Avg.'] * 5)
    })

    fig = px.line_polar(df_radar, r='Value', theta='Feature', color='Group', 
                        line_close=True, range_r=[0, 1],
                        color_discrete_map={
                            'Current Patient': patient_color, 
                            'Malignant Avg.': '#e4bdba', # Màu đỏ nhạt viền nét đứt trong thiết kế
                            'Benign Avg.': '#94a3b8' # Màu xám nhạt
                        })
    fig.update_traces(fill='toself', opacity=0.4)
    # Ẩn background biểu đồ cho sang trọng
    fig.update_layout(polar=dict(bgcolor="white"), paper_bgcolor="white") 
    
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 4. Trích xuất 3 Láng giềng gần nhất (Dành riêng cho UI)
    # Tính khoảng cách Euclid từ bệnh nhân này đến toàn bộ tập dữ liệu gốc
    distances = np.sqrt(np.sum((X_scaled_full - patient_scaled[0]) ** 2, axis=1))
    k_indices = np.argsort(distances)[:3] # Lấy 3 người giống nhất
    
    # Vẽ 3 Thẻ Metric (Cards)
    col1, col2, col3 = st.columns(3)
    
    neighbors = []
    for i in range(3):
        idx = k_indices[i]
        label_str = "Malignant" if y_raw[idx] == 1 else "Benign"
        dist_val = round(distances[idx], 3)
        neighbors.append((label_str, dist_val))

    with col1:
        with st.container(border=True):
            st.caption("NEIGHBOR #1")
            st.metric(label="Relative Distance", value=f"{neighbors[0][1]}", delta=neighbors[0][0], delta_color="inverse" if neighbors[0][0]=="Malignant" else "normal")
            
    with col2:
        with st.container(border=True):
            st.caption("NEIGHBOR #2")
            st.metric(label="Relative Distance", value=f"{neighbors[1][1]}", delta=neighbors[1][0], delta_color="inverse" if neighbors[1][0]=="Malignant" else "normal")

    with col3:
        with st.container(border=True):
            st.caption("NEIGHBOR #3")
            st.metric(label="Relative Distance", value=f"{neighbors[2][1]}", delta=neighbors[2][0], delta_color="inverse" if neighbors[2][0]=="Malignant" else "normal")

else:
    st.write("👈 Vui lòng nhập thông số lâm sàng bên thanh công cụ và nhấn **Predict Diagnosis**.")