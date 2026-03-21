import csv
import numpy as np

def load_data(filepath):
    """
    Đọc dữ liệu từ file CSV không dùng Pandas.
    Input: Đường dẫn file csv
    Output: 
        - X: Ma trận đặc trưng (Numpy array)
        - y: Nhãn (Numpy array)
    """
    features = []
    labels = []

    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        
        # Bỏ qua dòng tiêu đề (Header)
        header = next(reader) 
        
        for row in reader:
            if not row: continue # Bỏ qua dòng trống nếu có
            
            # Cấu trúc file data.csv:
            # Cột 0: ID (Bỏ qua)
            # Cột 1: Diagnosis (M/B) -> Cần mã hóa
            # Cột 2 đến 31: 30 đặc trưng số học -> Cần lấy
            # Cột 32: Unnamed (Rác) -> Bỏ qua
            
            # 1. Xử lý Nhãn (Cột 1)
            # Nếu là 'M' (Ác tính) -> 1, 'B' (Lành tính) -> 0
            label_str = row[1]
            if label_str == 'M':
                labels.append(1)
            else:
                labels.append(0)
            
            # 2. Xử lý Đặc trưng (Cột 2 đến 31)
            # Lấy từ index 2 đến 32 (không lấy 32)
            # Ép kiểu từ string sang float
            feature_row = [float(x) for x in row[2:32]] 
            features.append(feature_row)

    # Chuyển list thường thành Numpy Array để tính toán ma trận
    X = np.array(features)
    y = np.array(labels)
    
    return X, y

def min_max_scaler(X):
    """
    Chuẩn hóa dữ liệu về khoảng [0, 1] không dùng Sklearn.
    Công thức: (X - min) / (max - min)
    """
    # Tìm min, max dọc theo trục cột (axis=0)
    # Nghĩa là tìm min/max của từng đặc trưng (bán kính, diện tích...)
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    
    # Tránh chia cho 0 (nếu max == min)
    denominator = max_val - min_val
    denominator[denominator == 0] = 1.0 
    
    X_scaled = (X - min_val) / denominator
    return X_scaled

def train_test_split(X, y, test_size=0.2, random_seed=42):
    """
    Chia dữ liệu train/test ngẫu nhiên không dùng Sklearn.
    Input: X, y, tỉ lệ test (ví dụ 0.2 = 20%)
    """
    # Đặt hạt giống ngẫu nhiên để kết quả giống nhau mỗi lần chạy (quan trọng khi debug)
    np.random.seed(random_seed)
    
    # Tạo một danh sách chỉ mục (index) từ 0 đến n_samples
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples) # Tráo đổi ngẫu nhiên thứ tự
    
    # Xác định điểm cắt
    test_set_size = int(n_samples * test_size)
    
    # Cắt chỉ mục
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    
    # Chia dữ liệu theo chỉ mục đã trộn
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test