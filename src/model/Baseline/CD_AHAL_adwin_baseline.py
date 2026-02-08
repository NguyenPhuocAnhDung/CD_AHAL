import os
import time
import datetime
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend as K
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.utils import class_weight
from tabulate import tabulate

# Thư viện phát hiện Drift chuẩn (theo yêu cầu Reviewer 3 & Paper 1)
try:
    from river import drift
except ImportError:
    print("❌ Thư viện 'river' chưa được cài đặt. Vui lòng chạy: pip install river")
    exit()

# ==========================================
# 1. CẤU HÌNH CHO BASELINE "ADWIN"
# ==========================================
# Mục tiêu: So sánh CD-AHAL với ADWIN (Drift Detector cổ điển mạnh nhất)
DATA_PATH = "../../../dataset/processed_v3_unified"
BASE_OUTPUT = "../../../baocao/ADWIN_BASELINE_REVIEW3" # Folder riêng
MODEL_PATH = os.path.join(BASE_OUTPUT, "models")
REPORT_PATH = os.path.join(BASE_OUTPUT, "reports_stream")
PLOT_PATH = os.path.join(BASE_OUTPUT, "plots_stream")
SOURCE_MODEL_PATH = "../../../baocao/CD_AHAL_FINAL_FULL_METRICS_FINAL/models"

for p in [REPORT_PATH, PLOT_PATH]: 
    if not os.path.exists(p): os.makedirs(p)

ONLINE_STREAM_FILE = os.path.join(DATA_PATH, "processed_online_stream.parquet")

TIME_STEPS = 10
BATCH_SIZE = 512
RAW_BATCH_SIZE = BATCH_SIZE * TIME_STEPS
REHEARSAL_SIZE = 5000

# --- CẤU HÌNH ADWIN (Dựa theo Paper 1) ---
# ADWIN tự động điều chỉnh cửa sổ, không cần ngưỡng cố định.
# Nó giám sát "Loss" hoặc "Error Rate".

# --- CUSTOM LAYERS ---
@tf.keras.utils.register_keras_serializable()
class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs): super().__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='w', shape=(input_shape[-1], 1), initializer='normal')
        self.b = self.add_weight(name='b', shape=(input_shape[1], 1), initializer='zeros')
        super().build(input_shape)
    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b); a = K.softmax(e, axis=1)
        return K.sum(x * a, axis=1)

@tf.keras.utils.register_keras_serializable()
class MCDropout(layers.Dropout):
    def call(self, inputs, training=None): return super().call(inputs, training=True)

@tf.keras.utils.register_keras_serializable()
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision(); self.recall = tf.keras.metrics.Recall()
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
    def result(self):
        p = self.precision.result(); r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))
    def reset_state(self): self.precision.reset_state(); self.recall.reset_state()

@tf.keras.utils.register_keras_serializable()
class SparseFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs); self.gamma = gamma; self.alpha = alpha
        self.ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    def call(self, y_true, y_pred):
        ce = self.ce(y_true, y_pred); pt = tf.exp(-ce)
        return self.alpha * ((1 - pt) ** self.gamma) * ce
    def get_config(self): return {"gamma": self.gamma, "alpha": self.alpha}

# ==========================================
# HELPER
# ==========================================
class SmartBuffer:
    def __init__(self, max_size):
        self.max_size = max_size; self.buffer = [] 
    def add(self, x, y):
        for i in range(len(x)): self.buffer.append((x[i], y[i]))
        if len(self.buffer) > self.max_size:
            self.buffer = random.sample(self.buffer, self.max_size)
    def get_sample(self, n):
        if not self.buffer: return None, None
        s = random.sample(self.buffer, min(len(self.buffer), n))
        return np.array([x[0] for x in s]), np.array([x[1] for x in s])

def prepare_sequences(X, y, time_steps):
    if len(X) < time_steps: return np.array([]), np.array([])
    n = len(X) // time_steps
    X_seq = X[:n*time_steps].reshape((n, time_steps, X.shape[1]))
    y_seq = y[time_steps-1::time_steps]
    return X_seq, y_seq

def calc_all_metrics(y_true, y_pred, avg='weighted'):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average=avg, zero_division=0),
        'Recall': recall_score(y_true, y_pred, average=avg, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, average=avg, zero_division=0)
    }

def plot_visuals(batches, accs, retrain_indices, name):
    plt.figure(figsize=(14, 6))
    plt.plot(batches, accs, label='ADWIN Baseline Accuracy', color='darkorange')
    if retrain_indices:
        vals = [accs[b] for b in retrain_indices]
        plt.scatter(retrain_indices, vals, color='red', zorder=5, label='ADWIN Drift Detected', marker='x', s=60)
    plt.ylabel("Accuracy"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.title(f"Baseline: ADWIN Drift Detection & Retraining")
    plt.savefig(os.path.join(PLOT_PATH, f"{name}.png"), dpi=300); plt.close()

# ==========================================
# MAIN ADWIN BASELINE
# ==========================================
def main():
    print(f"--- BASELINE: ADWIN DRIFT DETECTOR (From Paper 1 & Reviewer 3) ---")
    custom = {'AttentionLayer': AttentionLayer, 'MCDropout': MCDropout, 'F1Score': F1Score, 'SparseFocalLoss': SparseFocalLoss}
    
    # Load model gốc
    try:
        model_A = keras.models.load_model(os.path.join(SOURCE_MODEL_PATH, "CD_AHAL_model_A_initial.h5"), custom_objects=custom)
        model_A.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print("✅ Initial Models loaded successfully!")
    except: 
        print(f"❌ Error loading models from {SOURCE_MODEL_PATH}. Check path!"); return

    df_stream = pd.read_parquet(ONLINE_STREAM_FILE)
    feat_cols = [c for c in df_stream.columns if c not in ['Label', 'Label_Multi', 'Label_Bin']]
    X_raw = df_stream[feat_cols].values; yb_raw = df_stream['Label'].values
    
    # Khởi tạo ADWIN Detector từ thư viện River
    adwin = drift.ADWIN(delta=0.002) # delta mặc định thường là 0.002
    
    buffer = SmartBuffer(REHEARSAL_SIZE)
    log = {'batch': [], 'acc': [], 'drift': []}
    
    n_batches = len(X_raw) // RAW_BATCH_SIZE
    all_true, all_pred = [], []
    
    print(f"-> Streaming {n_batches} batches with ADWIN...")
    
    for i in range(n_batches):
        s = i * RAW_BATCH_SIZE; e = (i+1) * RAW_BATCH_SIZE
        X_seq, y_cur = prepare_sequences(X_raw[s:e], yb_raw[s:e], TIME_STEPS)
        if len(X_seq) == 0: continue
        
        # 1. Inference
        preds_prob = model_A.predict(X_seq, verbose=0)
        preds = (preds_prob > 0.5).astype(int).flatten()
        acc = accuracy_score(y_cur, preds)
        
        all_true.extend(y_cur); all_pred.extend(preds)
        log['batch'].append(i); log['acc'].append(acc)
        
        # 2. ADWIN Drift Detection
        # ADWIN nhận vào từng dòng dữ liệu (0 hoặc 1). Ở đây ta đưa vào "Loss" hoặc "Error" (0: đúng, 1: sai)
        # Chúng ta sẽ đưa Error Rate của batch vào để check drift
        batch_error = 1.0 - acc
        adwin.update(batch_error) # Update ADWIN với lỗi của batch hiện tại
        
        if adwin.drift_detected:
            log['drift'].append(i)
            print(f"⚠️ ADWIN Drift Detected @ Batch {i} | Acc dropped to: {acc:.3f}")
            
            # 3. Retraining Strategy (Khi ADWIN báo động)
            # Reviewer 3 yêu cầu so sánh với baseline mạnh.
            # Ta dùng Random Sampling 20% (giống Periodic) nhưng chỉ kích hoạt khi ADWIN báo.
            budget_size = int(len(X_seq) * 0.20)
            idx_random = np.random.choice(len(X_seq), budget_size, replace=False)
            
            X_train_new = X_seq[idx_random]
            y_train_new = y_cur[idx_random]
            
            # Rehearsal Buffer
            X_buf, y_buf = buffer.get_sample(BATCH_SIZE)
            if X_buf is not None:
                X_train_new = np.concatenate([X_train_new, X_buf], axis=0)
                y_train_new = np.concatenate([y_train_new, y_buf], axis=0)
            
            # Update Model
            if len(X_train_new) > 0:
                cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_new), y=y_train_new)
                cw_d = dict(enumerate(cw))
                model_A.fit(X_train_new, y_train_new, epochs=3, batch_size=64, verbose=0, class_weight=cw_d)
            
            # Reset ADWIN (Optional, but often good practice after adaptation)
            # adwin = drift.ADWIN(delta=0.002) 
            
            buffer.add(X_seq[idx_random], y_cur[idx_random])
        
        if i % 20 == 0: print(f"Batch {i}/{n_batches} | Acc: {acc:.3f}")

    # Summary
    print("\n--- ADWIN BASELINE SUMMARY ---")
    plot_visuals(log['batch'], log['acc'], log['drift'], "ADWIN_Retrain_Performance")
    
    m_bin = calc_all_metrics(all_true, all_pred)
    df_res = pd.DataFrame([m_bin], index=['ADWIN Baseline'])
    print(tabulate(df_res, headers='keys', tablefmt='grid'))
    
    df_res.to_csv(os.path.join(REPORT_PATH, "ADWIN_Baseline_Metrics.csv"))
    print(f"✅ Done. Results saved to {REPORT_PATH}")

if __name__ == "__main__":
    main()