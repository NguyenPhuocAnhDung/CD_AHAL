import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend as K
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import class_weight
from tabulate import tabulate
import random

# ==========================================
# 1. CẤU HÌNH
# ==========================================
DATA_PATH = "../../../dataset/processed_v3_unified"
BASE_OUTPUT = "../../../baocao/TRANSFORMER_BASELINE_REVIEW3"
MODEL_PATH = os.path.join(BASE_OUTPUT, "models")
REPORT_PATH = os.path.join(BASE_OUTPUT, "reports_stream")
PLOT_PATH = os.path.join(BASE_OUTPUT, "plots_stream")

for p in [REPORT_PATH, PLOT_PATH, MODEL_PATH]: 
    if not os.path.exists(p): os.makedirs(p)

ONLINE_STREAM_FILE = os.path.join(DATA_PATH, "processed_online_stream.parquet")
TIME_STEPS = 10
BATCH_SIZE = 512
RAW_BATCH_SIZE = BATCH_SIZE * TIME_STEPS

# ==========================================
# 2. XÂY DỰNG MODEL TRANSFORMER (Trend 2025)
# ==========================================
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Self Attention
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

def build_transformer_model(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    # Transformer Block
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)
    # Global Pooling
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def prepare_sequences(X, y, time_steps):
    if len(X) < time_steps: return np.array([]), np.array([])
    n = len(X) // time_steps
    X_seq = X[:n*time_steps].reshape((n, time_steps, X.shape[1]))
    y_seq = y[time_steps-1::time_steps]
    return X_seq, y_seq

def main():
    print("--- BASELINE: TRANSFORMER (SOTA 2025 COMPARISON) ---")
    
    # Load Data
    print("-> Loading Data...")
    if not os.path.exists(ONLINE_STREAM_FILE):
        print("❌ Data file not found!"); return
        
    df_stream = pd.read_parquet(ONLINE_STREAM_FILE)
    feat_cols = [c for c in df_stream.columns if c not in ['Label', 'Label_Multi', 'Label_Bin']]
    X_raw = df_stream[feat_cols].values
    yb_raw = df_stream['Label'].values
    
    # Build & Train Initial Transformer (Giả lập đã train Phase 1)
    # Trong thực tế, ta nên load weight, nhưng ở đây ta build mới và train nhanh 1 đoạn đầu
    # để mô phỏng một model Transformer đã học.
    print("-> Building Transformer Model...")
    model = build_transformer_model((TIME_STEPS, X_raw.shape[1]))
    
    # Warm-up training (Dùng 20% dữ liệu đầu để train nhanh cho model khôn ra)
    warmup_size = int(len(X_raw) * 0.1) 
    X_warm, y_warm = prepare_sequences(X_raw[:warmup_size], yb_raw[:warmup_size], TIME_STEPS)
    model.fit(X_warm, y_warm, epochs=2, batch_size=256, verbose=1)
    print("✅ Transformer Warm-up Complete. Starting Stream...")

    # Streaming Phase
    log = {'batch': [], 'acc': []}
    n_batches = (len(X_raw) - warmup_size) // RAW_BATCH_SIZE
    
    # Simulate Periodic Retraining for Transformer (Fair comparison)
    # Transformer cũng được train lại định kỳ để so sánh sức mạnh kiến trúc
    
    all_true, all_pred = [], []
    
    for i in range(n_batches):
        s = warmup_size + i * RAW_BATCH_SIZE
        e = warmup_size + (i+1) * RAW_BATCH_SIZE
        X_seq, y_cur = prepare_sequences(X_raw[s:e], yb_raw[s:e], TIME_STEPS)
        if len(X_seq) == 0: continue
        
        # Inference
        preds = (model.predict(X_seq, verbose=0) > 0.5).astype(int).flatten()
        acc = accuracy_score(y_cur, preds)
        
        all_true.extend(y_cur); all_pred.extend(preds)
        log['batch'].append(i); log['acc'].append(acc)
        
        # Periodic Retraining (Every 20 batches)
        if i > 0 and i % 20 == 0:
            # Simple retraining on current batch
            model.fit(X_seq, y_cur, epochs=1, batch_size=64, verbose=0)
            print(f"🔄 Transformer Retrain @ {i} | Acc: {acc:.3f}")

        if i % 20 == 0: print(f"Batch {i}/{n_batches} | Transformer Acc: {acc:.3f}")

    # Save Results
    pd.DataFrame(log).to_csv(os.path.join(REPORT_PATH, "Transformer_Metrics.csv"), index=False)
    
    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(log['batch'], log['acc'], label='Transformer (Periodic)', color='blue')
    plt.title("Baseline: Transformer Network (SOTA Architecture)")
    plt.ylabel("Accuracy"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOT_PATH, "Transformer_Performance.png"))
    
    print("✅ Done.")

if __name__ == "__main__":
    main()