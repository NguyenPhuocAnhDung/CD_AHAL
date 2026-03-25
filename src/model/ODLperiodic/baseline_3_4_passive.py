import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend as K
# [UPDATE 1] Thêm các metrics cần thiết
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ==========================================
# CẤU HÌNH ĐỒNG BỘ
# ==========================================
DATA_PATH = "../../../dataset/processed_v3_unified"
MODEL_PATH = "../../../baocao/CD_AHAL_FINAL_FULL_METRICS_FINAL/models/CD_AHAL_model_A_initial.h5"
OUTPUT_PATH = "../../../baocao/BASELINES_RESULT"
TARGET_BATCHES = 249 

if not os.path.exists(OUTPUT_PATH): os.makedirs(OUTPUT_PATH)

# --- Custom Layers (Giữ nguyên) ---
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
    def call(self, inputs): return super().call(inputs, training=True)

# --- Run Function ---
def run_passive_baseline(mode="ODL"):
    print(f"\n>>> RUNNING BASELINE: {mode} (Limit: {TARGET_BATCHES})")
    
    # 1. Load Data & Model
    try:
        # Giả định file parquet đã có sẵn
        df = pd.read_parquet(os.path.join(DATA_PATH, "processed_online_stream.parquet"))
        feat_cols = [c for c in df.columns if c not in ['Label', 'Label_Multi', 'Label_Bin']]
        X_raw = df[feat_cols].values
        y_raw = df['Label'].values
        
        model = keras.models.load_model(MODEL_PATH, custom_objects={'AttentionLayer': AttentionLayer, 'MCDropout': MCDropout})
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    except Exception as e: print(f"❌ Lỗi: {e}"); return

    BATCH_SIZE = 512
    TIME_STEPS = 10
    RAW_BATCH_SIZE = BATCH_SIZE * TIME_STEPS
    n_features = X_raw.shape[1]
    
    log = []
    
    # 2. Run Loop
    for i in range(TARGET_BATCHES + 1):
        s = i * RAW_BATCH_SIZE
        e = (i + 1) * RAW_BATCH_SIZE
        if e > len(X_raw): break
        
        X_seq = X_raw[s:e].reshape((BATCH_SIZE, TIME_STEPS, n_features))
        # Lấy nhãn của time step cuối cùng trong chuỗi
        y_seq = y_raw[s:e][TIME_STEPS-1::TIME_STEPS] 
        
        # Inference
        t0 = time.time()
        preds = model.predict(X_seq, verbose=0)
        inf_time = (time.time() - t0) * 1000
        
        # [UPDATE 2] Tính toán 4 chỉ số
        # Chuyển xác suất thành nhãn nhị phân (0 hoặc 1)
        y_pred_bin = (preds > 0.5).astype(int).flatten()
        
        acc = accuracy_score(y_seq, y_pred_bin)
        # zero_division=0 để tránh lỗi nếu batch không có mẫu dương
        pre = precision_score(y_seq, y_pred_bin, zero_division=0)
        rec = recall_score(y_seq, y_pred_bin, zero_division=0)
        f1 = f1_score(y_seq, y_pred_bin, zero_division=0)
        
        # Logic Train
        should_train = False
        status = "Normal"
        
        if mode == "ODL":
            should_train = True
            status = "ODL Update"
        elif mode == "PERIODIC":
            if i > 0 and i % 20 == 0:
                should_train = True
                status = "Periodic Update"
        
        train_time = 0
        if should_train:
            t1 = time.time()
            model.fit(X_seq, y_seq, epochs=1, batch_size=64, verbose=0)
            train_time = (time.time() - t1) * 1000
            
        # [UPDATE 3] Log đầy đủ 4 chỉ số
        log.append([i, acc, pre, rec, f1, inf_time, train_time, status])
        
        if i % 20 == 0: 
            print(f"Batch {i}/{TARGET_BATCHES} | Mode: {mode} | Acc: {acc:.4f} | F1: {f1:.4f}")

    # [UPDATE 4] Lưu file CSV với header mới
    file_name = f"Baseline_{mode}.csv"
    columns = ['Batch', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Inf_Time_ms', 'Train_Time_ms', 'Status']
    pd.DataFrame(log, columns=columns).to_csv(os.path.join(OUTPUT_PATH, file_name), index=False)
    print(f"✅ Done {mode}. Results saved to {file_name}")

if __name__ == "__main__":
    run_passive_baseline(mode="ODL")
    run_passive_baseline(mode="PERIODIC")