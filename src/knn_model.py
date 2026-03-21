import numpy as np
from collections import Counter

class KNN_Classifier:
    def __init__(self, k=3):
        """
        Khởi tạo mô hình KNN.
        k: số lượng láng giềng gần nhất cần xét.
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """
        Huấn luyện mô hình (Với KNN thì chỉ cần lưu trữ dữ liệu).
        """
        self.X_train = X_train
        self.y_train = y_train

    def _euclidean_distance(self, x_test_point):
        """
        [TỐI ƯU HÓA] Tính khoảng cách từ 1 điểm test tới TOÀN BỘ tập train cùng lúc.
        Sử dụng Vector hóa (Broadcasting) của NumPy thay vì vòng lặp for.
        """
        # x_test_point được trừ đi toàn bộ ma trận X_train
        distances = np.sqrt(np.sum((self.X_train - x_test_point) ** 2, axis=1))
        return distances

    def predict(self, X_test):
        """
        Dự đoán nhãn cho một tập dữ liệu test.
        """
        # Áp dụng hàm _predict_one cho từng dòng dữ liệu trong X_test
        predictions = [self._predict_one(x) for x in X_test]
        return np.array(predictions)

    def _predict_one(self, x):
        """
        Dự đoán cho MỘT bệnh nhân duy nhất.
        """
        # 1. Tính khoảng cách
        distances = self._euclidean_distance(x)
        
        # 2. Tìm k láng giềng gần nhất (lấy index)
        k_indices = np.argsort(distances)[:self.k]
        
        # 3. Lấy nhãn của k láng giềng đó
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # 4. Bầu cử theo số đông (Majority vote)
        most_common = Counter(k_nearest_labels).most_common(1)
        
        return most_common[0][0]