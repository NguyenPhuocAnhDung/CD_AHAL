import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- CẤU HÌNH ĐƯỜNG DẪN CHÍNH XÁC (ĐÃ SỬA) ---
# File đang ở: .../CD_AHAL/src/model/CD_AHAL/plot_latency_cdahal.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Lùi 3 cấp để về Root: .../CD_AHAL/
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../"))

# Đường dẫn đến log
LOG_PATH = os.path.join(PROJECT_ROOT, "baocao/CD_AHAL_FINAL_FULL_METRICS_FINAL/reports_stream/Drift_Log.csv")
OUTPUT_IMG_PATH = os.path.join(PROJECT_ROOT, "baocao/IMAGES")

if not os.path.exists(OUTPUT_IMG_PATH):
    os.makedirs(OUTPUT_IMG_PATH)

def main():
    print(f"📂 Đang đọc log từ: {LOG_PATH}")
    
    if not os.path.exists(LOG_PATH):
        print(f"❌ Lỗi: Không tìm thấy file tại đường dẫn trên!")
        print("👉 Hãy kiểm tra xem file Drift_Log.csv có thực sự nằm ở đó không.")
        return

    try:
        df = pd.read_csv(LOG_PATH)
        print(f"✅ Đã load {len(df)} sự kiện Drift.")
    except Exception as e:
        print(f"❌ Lỗi đọc file: {e}")
        return

    # Tính toán thống kê
    avg_lat = df['Latency(ms)'].mean()
    med_lat = df['Latency(ms)'].median()
    max_lat = df['Latency(ms)'].max()
    min_lat = df['Latency(ms)'].min()

    # Style
    sns.set_style("whitegrid")
    
    # ---------------------------------------------------------
    # BIỂU ĐỒ 1: TIMELINE
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    
    plt.plot(df['Batch'], df['Latency(ms)'], color='#1f77b4', linestyle='-', linewidth=1.5, alpha=0.5, label='Latency Trend')
    scatter = plt.scatter(df['Batch'], df['Latency(ms)'], c=df['Latency(ms)'], cmap='viridis', s=80, edgecolor='k', zorder=10)
    
    plt.axhline(y=avg_lat, color='#d62728', linestyle='--', linewidth=2, label=f'Avg: {avg_lat:.0f} ms')

    plt.colorbar(scatter, label='Latency (ms)')
    plt.title("CD-AHAL Adaptation Latency Analysis", fontsize=16, fontweight='bold')
    plt.xlabel("Drift Event (Batch Index)", fontsize=12)
    plt.ylabel("Retraining Latency (ms)", fontsize=12)
    plt.legend(loc='upper right')
    
    save_path1 = os.path.join(OUTPUT_IMG_PATH, "CD_AHAL_Latency_Timeline.png")
    plt.savefig(save_path1, dpi=300, bbox_inches='tight')
    print(f"✅ Đã lưu Timeline: {save_path1}")

    # ---------------------------------------------------------
    # BIỂU ĐỒ 2: DISTRIBUTION
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    sns.histplot(df['Latency(ms)'], kde=True, bins=15, color='#2ca02c', edgecolor='black')
    plt.axvline(x=avg_lat, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_lat:.0f}ms')
    
    plt.title("Distribution of Adaptation Latency", fontsize=16, fontweight='bold')
    plt.xlabel("Latency (ms)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    
    save_path2 = os.path.join(OUTPUT_IMG_PATH, "CD_AHAL_Latency_Distribution.png")
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    print(f"✅ Đã lưu Distribution: {save_path2}")

if __name__ == "__main__":
    main()