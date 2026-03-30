import os
import sys
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.decomposition import PCA

# Import existing backend
from data_loader import load_data, min_max_scaler
from knn_model import KNN_Classifier

app = Flask(__name__, template_folder='../templates')

# --- INITIALIZE SYSTEM ---
X_raw, y = None, None
X_scaled = None
min_vals, max_vals = None, None
model = None
m_avg_scaled, b_avg_scaled = None, None
pca_model = None
X_pca = None
corr_matrix = None

def init_system():
    global X_raw, y, X_scaled, min_vals, max_vals, model, m_avg_scaled, b_avg_scaled, pca_model, X_pca, corr_matrix
    try:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data.csv')
        X_raw, y = load_data(data_path)
        min_vals = np.min(X_raw, axis=0)
        max_vals = np.max(X_raw, axis=0)
        
        X_scaled = min_max_scaler(X_raw)
        model = KNN_Classifier(k=5)
        model.fit(X_scaled, y)
        
        m_avg_scaled = (np.mean(X_raw[y == 1], axis=0) - min_vals) / (max_vals - min_vals + 1e-9)
        b_avg_scaled = (np.mean(X_raw[y == 0], axis=0) - min_vals) / (max_vals - min_vals + 1e-9)
        
        # PCA preparation
        pca_model = PCA(n_components=2)
        X_pca = pca_model.fit_transform(X_scaled)

        # Correlation computation
        y_col = y.reshape(-1, 1)
        full_data = np.hstack((X_scaled, y_col))
        corr_matrix = np.corrcoef(full_data, rowvar=False)

        print("System Initialized successfully!")
    except Exception as e:
        print(f"Error initializing: {e}")

init_system()

@app.route('/chart_data', methods=['GET'])
def chart_data():
    malignant_points = [{'x': float(X_raw[i, 0]), 'y': float(X_raw[i, 1])} for i in range(len(y)) if y[i] == 1]
    benign_points = [{'x': float(X_raw[i, 0]), 'y': float(X_raw[i, 1])} for i in range(len(y)) if y[i] == 0]
    
    # PCA Points
    pca_malignant = [{'x': float(X_pca[i, 0]), 'y': float(X_pca[i, 1])} for i in range(len(y)) if y[i] == 1]
    pca_benign = [{'x': float(X_pca[i, 0]), 'y': float(X_pca[i, 1])} for i in range(len(y)) if y[i] == 0]

    # Correlation Matrix (Top 10 features for simplicity in visualization + 1 Target)
    idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 30] # 10 means + y
    small_corr = corr_matrix[np.ix_(idx, idx)].tolist()
    
    m_avg_mean = np.mean(X_raw[y == 1][:, :10], axis=0).tolist()
    b_avg_mean = np.mean(X_raw[y == 0][:, :10], axis=0).tolist()
    
    return jsonify({
        'scatter': { 'malignant': malignant_points, 'benign': benign_points },
        'bar': { 'malignant': m_avg_mean, 'benign': b_avg_mean },
        'pca': { 'malignant': pca_malignant, 'benign': pca_benign },
        'corr': small_corr
    })

@app.route('/')
def index():
    feature_names = ["Radius", "Texture", "Perimeter", "Area", "Smoothness", "Compactness", "Concavity", "Concave Points", "Symmetry", "Fractal Dim"]
    
    # Defaults
    default_vals = np.mean(X_raw, axis=0).tolist() if X_raw is not None else []
    min_v = min_vals.tolist() if min_vals is not None else []
    max_v = max_vals.tolist() if max_vals is not None else []
    
    # We will pass structured data to the template
    return render_template('index.html', 
                           feature_names=feature_names, 
                           min_vals=min_v,
                           max_vals=max_v,
                           default_vals=default_vals)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features'], dtype=float)
    
    # Normalize
    patient_scaled = (features - min_vals) / (max_vals - min_vals + 1e-9)
    
    # Predict
    prediction = int(model.predict(patient_scaled.reshape(1, -1))[0])
    
    # Radar chart data: Radius(0), Texture(1), Area(3), Concavity(6), Smoothness(4)
    idx = [0, 1, 3, 6, 4]
    radar_patient = patient_scaled[idx].tolist()
    radar_m_avg = m_avg_scaled[idx].tolist()
    radar_b_avg = b_avg_scaled[idx].tolist()
    
    # Neighbors
    distances = np.sqrt(np.sum((X_scaled - patient_scaled) ** 2, axis=1))
    k_indices = np.argsort(distances)[:3]
    
    # Calculate patient PCA position
    patient_pca = pca_model.transform(patient_scaled.reshape(1, -1))[0]
    
    neighbors = []
    for i in k_indices:
        neighbors.append({
            'distance': round(float(distances[i]), 3),
            'label': int(y[i])
        })
        
    return jsonify({
        'prediction': prediction,
        'radar': {
            'patient': radar_patient,
            'malignant_avg': radar_m_avg,
            'benign_avg': radar_b_avg
        },
        'patient_pca': [float(patient_pca[0]), float(patient_pca[1])],
        'neighbors': neighbors
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
