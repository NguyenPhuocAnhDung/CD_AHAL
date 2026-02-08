import os, time, pandas as pd
from river import forest, stream, metrics

# ==========================================
# CẤU HÌNH ĐỒNG BỘ
# ==========================================
DATA_PATH = "../../../dataset/processed_v3_unified"
OUTPUT_PATH = "../../../baocao/BASELINES_RESULT"
TARGET_BATCHES = 247 # <--- KHỚP VỚI CD-AHAL
BATCH_SIZE = 512

if not os.path.exists(OUTPUT_PATH): os.makedirs(OUTPUT_PATH)

def main():
    print(f">>> RUNNING BASELINE 2: ARF (Limit: {TARGET_BATCHES})")
    
    try:
        df = pd.read_parquet(os.path.join(DATA_PATH, "processed_online_stream.parquet"))
        X = df.drop(columns=['Label', 'Label_Multi', 'Label_Bin'])
        y = df['Label']
    except: print("❌ Lỗi data!"); return

    model = forest.ARFClassifier(n_models=10, seed=42)
    metric = metrics.Accuracy()
    
    log = []
    batch_time = 0
    batch_cnt = 0
    
    for i, (x, y_true) in enumerate(stream.iter_pandas(X, y)):
        t0 = time.time()
        y_pred = model.predict_one(x)
        model.learn_one(x, y_true)
        batch_time += (time.time() - t0)
        
        if y_pred is not None: metric.update(y_true, y_pred)
        
        # Check Batch Boundary (mỗi 512 mẫu = 1 batch của Deep Learning)
        if (i + 1) % BATCH_SIZE == 0:
            log.append([batch_cnt, metric.get(), batch_time * 1000])
            batch_time = 0
            
            if batch_cnt % 20 == 0: print(f"Batch {batch_cnt}/{TARGET_BATCHES}: Acc {metric.get():.4f}")
            
            # Dừng đúng lúc
            if batch_cnt >= TARGET_BATCHES:
                break
            batch_cnt += 1

    pd.DataFrame(log, columns=['Batch', 'Accuracy', 'Total_Time_ms']).to_csv(os.path.join(OUTPUT_PATH, "Baseline_ARF.csv"), index=False)
    print("✅ Done Baseline 2.")

if __name__ == "__main__": main()