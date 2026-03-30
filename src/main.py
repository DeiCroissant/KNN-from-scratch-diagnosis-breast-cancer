import sys
import numpy as np

# Thiết lập encoding UTF-8 để in tiếng Việt ra terminal không bị lỗi trên Windows
sys.stdout.reconfigure(encoding='utf-8')

from data_loader import load_data, min_max_scaler, train_test_split
from knn_model import KNN_Classifier

def calculate_metrics(y_true, y_pred):
    """D
    Tự code hàm tính toán các chỉ số đánh giá (Từ ma trận nhầm lẫn - Confusion Matrix)
    1: Ác tính (Malignant) - Positive
    0: Lành tính (Benign) - Negative
    """
    # Tính các thành phần của Confusion Matrix
    TP = np.sum((y_true == 1) & (y_pred == 1)) # True Positive: Đoán Ác tính, Thật sự Ác tính
    TN = np.sum((y_true == 0) & (y_pred == 0)) # True Negative: Đoán Lành tính, Thật sự Lành tính
    FP = np.sum((y_true == 0) & (y_pred == 1)) # False Positive: Đoán Ác tính, Thật ra Lành tính (Báo động giả)
    FN = np.sum((y_true == 1) & (y_pred == 0)) # False Negative: Đoán Lành tính, Thật ra Ác tính (Bỏ sót - CỰC KỲ NGUY HIỂM)

    # Tính toán 4 chỉ số
    total = len(y_true)
    accuracy = (TP + TN) / total
    
    # Thêm 1e-9 để tránh lỗi chia cho 0
    precision = TP / (TP + FP + 1e-9) 
    recall = TP / (TP + FN + 1e-9)    
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)

    return {
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1-Score": float(f1_score),
        "Confusion Matrix": {"TP": int(TP), "TN": int(TN), "FP": int(FP), "FN": int(FN)}
    }

def run_evaluation():
    print("1. Đang tải và tiền xử lý dữ liệu...")
    # Cập nhật đường dẫn file csv của bạn nếu cần
    X, y = load_data('data.csv') 
    X_scaled = min_max_scaler(X)
    
    print("2. Đang chia tập Train/Test (Tỷ lệ 80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_seed=42)
    
    # Khởi tạo K = 5 (Bạn có thể thay đổi số này)
    k_value = 5
    print(f"3. Đang khởi tạo mô hình KNN với K = {k_value}...")
    model = KNN_Classifier(k=k_value)
    
    print("4. Đang huấn luyện (Fit) mô hình...")
    model.fit(X_train, y_train)
    
    print("5. Đang dự đoán trên tập Test...")
    y_pred, _ = model.predict(X_test)
    
    print("\n--- KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ---")
    results = calculate_metrics(y_test, y_pred)
    
    print(f"Độ chính xác tổng thể (Accuracy) : {results['Accuracy']*100:.2f}%")
    print(f"Độ chuẩn xác (Precision)        : {results['Precision']*100:.2f}%")
    print(f"Độ nhạy - Không bỏ sót (Recall) : {results['Recall']*100:.2f}%")
    print(f"Điểm cân bằng (F1-Score)        : {results['F1-Score']*100:.2f}%")
    
    print("\n--- MA TRẬN NHẦM LẪN (Confusion Matrix) ---")
    print(f"Bắt trúng ung thư (TP)  : {results['Confusion Matrix']['TP']} ca")
    print(f"Đoán đúng khỏe mạnh (TN): {results['Confusion Matrix']['TN']} ca")
    print(f"Báo động giả (FP)       : {results['Confusion Matrix']['FP']} ca")
    print(f"BỎ SÓT UNG THƯ (FN)     : {results['Confusion Matrix']['FN']} ca")

if __name__ == "__main__":
    run_evaluation()