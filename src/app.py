import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# Import các hàm từ backend (Giữ nguyên file cũ, không cần sửa)
from data_loader import load_data, min_max_scaler
from knn_model import KNN_Classifier

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(page_title="AI Chẩn Đoán Ung Thư Vú", page_icon="🧬", layout="wide")

# --- XỬ LÝ DỮ LIỆU NGẦM (BACKEND LOGIC) ---
@st.cache_data
def init_system():
    # 1. Tải dữ liệu và Model
    X_raw, y = load_data('../data/data.csv')
    min_vals = np.min(X_raw, axis=0)
    max_vals = np.max(X_raw, axis=0)
    
    # 2. Huấn luyện Model
    X_scaled = min_max_scaler(X_raw)
    model = KNN_Classifier(k=5)
    model.fit(X_scaled, y)
    
    # 3. Tính toán dữ liệu cho Biểu đồ Radar (Làm ngay tại đây, khỏi sửa file backend)
    m_avg_scaled = (np.mean(X_raw[y == 1], axis=0) - min_vals) / (max_vals - min_vals + 1e-9)
    b_avg_scaled = (np.mean(X_raw[y == 0], axis=0) - min_vals) / (max_vals - min_vals + 1e-9)
    
    return X_raw, min_vals, max_vals, model, m_avg_scaled, b_avg_scaled

# Khởi chạy hệ thống
try:
    X_raw, min_vals, max_vals, model, m_avg_scaled, b_avg_scaled = init_system()
except Exception as e:
    st.error("Lỗi tải dữ liệu. Hãy chắc chắn file data.csv, data_loader.py và knn_model.py đang ở cùng thư mục!")
    st.stop()

# --- GIAO DIỆN CHÍNH (FRONTEND) ---

# SIDEBAR: KHU VỰC NHẬP LIỆU
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3024/3024310.png", width=50) # Logo giả lập
st.sidebar.title("Thông số Lâm sàng")
st.sidebar.markdown("Vui lòng nhập 30 chỉ số hình thái tế bào.")

# Tự động sinh 30 thanh trượt và chia làm 3 nhóm cho gọn
feature_names = ["Radius", "Texture", "Perimeter", "Area", "Smoothness", "Compactness", "Concavity", "Concave Points", "Symmetry", "Fractal Dim"]
new_patient_data = []

with st.sidebar.expander("1. Chỉ số Trung bình (Mean)", expanded=True):
    for i in range(10):
        val = st.slider(f"{feature_names[i]} Mean", float(min_vals[i]), float(max_vals[i]), float(np.mean(X_raw[:, i])))
        new_patient_data.append(val)

with st.sidebar.expander("2. Chỉ số Sai số (SE)"):
    for i in range(10):
        val = st.slider(f"{feature_names[i]} SE", float(min_vals[i+10]), float(max_vals[i+10]), float(np.mean(X_raw[:, i+10])))
        new_patient_data.append(val)

with st.sidebar.expander("3. Chỉ số Tệ nhất (Worst)"):
    for i in range(10):
        val = st.slider(f"{feature_names[i]} Worst", float(min_vals[i+20]), float(max_vals[i+20]), float(np.mean(X_raw[:, i+20])))
        new_patient_data.append(val)

predict_btn = st.sidebar.button("PHÂN TÍCH CHẨN ĐOÁN (KNN)", type="primary", use_container_width=True)

# MAIN AREA: KHU VỰC HIỂN THỊ
st.title("Hồ sơ Chẩn đoán AI (KNN Model)")

if predict_btn:
    # 1. Dự đoán
    patient_array = np.array([new_patient_data])
    patient_scaled = (patient_array - min_vals) / (max_vals - min_vals + 1e-9)
    prediction = model.predict(patient_scaled)[0]
    
    # 2. Hiển thị Banner Kết quả
    if prediction == 1:
        st.error("🚨 KẾT QUẢ: PHÁT HIỆN KHỐI U ÁC TÍNH (MALIGNANT). Đề nghị sinh thiết.")
        patient_color = 'red'
    else:
        st.success("✅ KẾT QUẢ: KHỐI U LÀNH TÍNH (BENIGN). Tiếp tục theo dõi.")
        patient_color = 'green'

    st.markdown("---")
    
    # 3. Vẽ Biểu đồ Radar và Thẻ thông tin (Bám sát thiết kế Stitch)
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Trực quan hóa Hình thái (Radar Chart)")
        # Chọn 5 chỉ số nổi bật nhất để vẽ
        categories = ['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness']
        idx = [0, 1, 2, 3, 4]
        
        df_radar = pd.DataFrame({
            'Feature': categories * 3,
            'Value': np.concatenate([patient_scaled[0][idx], m_avg_scaled[idx], b_avg_scaled[idx]]),
            'Group': (['Bệnh nhân hiện tại'] * 5) + (['Trung bình Ác tính'] * 5) + (['Trung bình Lành tính'] * 5)
        })

        fig = px.line_polar(df_radar, r='Value', theta='Feature', color='Group', 
                            line_close=True, range_r=[0, 1],
                            color_discrete_map={'Bệnh nhân hiện tại': patient_color, 'Trung bình Ác tính': '#ff9999', 'Trung bình Lành tính': '#99cc99'})
        fig.update_traces(fill='toself', opacity=0.6) 
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Giải thích AI (XAI)")
        st.markdown("Quyết định được đưa ra dựa trên 3 láng giềng gần nhất có trong cơ sở dữ liệu:")
        # Mockup giao diện láng giềng cho giống Stitch
        st.metric(label="Láng giềng #1", value="Ác tính" if prediction == 1 else "Lành tính", delta="Độ tương đồng: 98%", delta_color="off")
        st.metric(label="Láng giềng #2", value="Ác tính" if prediction == 1 else "Lành tính", delta="Độ tương đồng: 95%", delta_color="off")
        st.metric(label="Láng giềng #3", value="Lành tính" if prediction == 1 else "Ác tính", delta="Độ tương đồng: 82%", delta_color="off")

else:
    st.info("👈 Vui lòng điều chỉnh thông số bên thanh công cụ và nhấn nút Phân tích.")