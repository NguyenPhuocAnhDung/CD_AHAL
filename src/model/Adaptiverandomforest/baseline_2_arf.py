import os
import time
import pandas as pd
from river import forest, stream
# [UPDATE 1] Import sklearn để tính metrics theo từng batch (giống file Keras)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ==========================================
# CẤU HÌNH ĐỒNG BỘ
# ==========================================
DATA_PATH = "../../../dataset/processed_v3_unified"
OUTPUT_PATH = "../../../baocao/BASELINES_RESULT"
TARGET_BATCHES = 249 # <--- KHỚP VỚI CD-AHAL
BATCH_SIZE = 512

if not os.path.exists(OUTPUT_PATH): os.makedirs(OUTPUT_PATH)

def main():
    print(f">>> RUNNING BASELINE 2: ARF (Limit: {TARGET_BATCHES})")
    
    try:
        # Load dữ liệu
        df = pd.read_parquet(os.path.join(DATA_PATH, "processed_online_stream.parquet"))
        X = df.drop(columns=['Label', 'Label_Multi', 'Label_Bin'])
        y = df['Label']
    except Exception as e: 
        print(f"❌ Lỗi data: {e}")
        return

    # Khởi tạo mô hình ARF
    model = forest.ARFClassifier(n_models=10, seed=42)
    
    log = []
    
    # Biến tạm để gom dữ liệu trong 1 batch
    batch_y_true = []
    batch_y_pred = []
    
    batch_time = 0
    batch_cnt = 0
    
    # Stream dữ liệu
    for i, (x, y_true) in enumerate(stream.iter_pandas(X, y)):
        t0 = time.time()
        
        # 1. Dự đoán & Học (Online Learning)
        y_pred = model.predict_one(x)
        model.learn_one(x, y_true)
        
        batch_time += (time.time() - t0)
        
        # [UPDATE 2] Gom kết quả vào list tạm để tính metrics cuối batch
        # Nếu mô hình chưa predict được (lúc đầu), gán tạm là 0 hoặc nhãn mặc định
        pred_val = y_pred if y_pred is not None else 0 
        batch_y_true.append(y_true)
        batch_y_pred.append(pred_val)
        
        # Check Batch Boundary (mỗi 512 mẫu)
        if (i + 1) % BATCH_SIZE == 0:
            # [UPDATE 3] Tính 4 chỉ số bằng sklearn cho batch hiện tại
            acc = accuracy_score(batch_y_true, batch_y_pred)
            pre = precision_score(batch_y_true, batch_y_pred, zero_division=0)
            rec = recall_score(batch_y_true, batch_y_pred, zero_division=0)
            f1 = f1_score(batch_y_true, batch_y_pred, zero_division=0)
            
            # Ghi log: Batch, 4 Metrics, Time
            log.append([batch_cnt, acc, pre, rec, f1, batch_time * 1000])
            
            if batch_cnt % 20 == 0: 
                print(f"Batch {batch_cnt}/{TARGET_BATCHES}: Acc {acc:.4f} | F1 {f1:.4f}")
            
            # Reset biến cho batch mới
            batch_time = 0
            batch_y_true = []
            batch_y_pred = []
            
            # Dừng đúng lúc
            if batch_cnt >= TARGET_BATCHES:
                break
            batch_cnt += 1

    # [UPDATE 4] Lưu file CSV với đầy đủ cột
    cols = ['Batch', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Total_Time_ms']
    pd.DataFrame(log, columns=cols).to_csv(os.path.join(OUTPUT_PATH, "Baseline_ARF.csv"), index=False)
    print("✅ Done Baseline 2: ARF (Full Metrics).")

if __name__ == "__main__": 
    main()