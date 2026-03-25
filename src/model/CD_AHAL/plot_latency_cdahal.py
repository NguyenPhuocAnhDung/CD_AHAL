import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- CẤU HÌNH ĐƯỜNG DẪN (QUAN TRỌNG) ---
# File đang ở: .../src/model/CD_AHAL/plot_latency_cdahal.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Lùi 3 cấp để về Root dự án (D:/.../CD_AHAL/)
# (Trước đây là ../../ nên bị kẹt ở src/)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../"))

# Đường dẫn file Log
LOG_PATH = os.path.join(PROJECT_ROOT, "baocao/CD_AHAL_FINAL_FULL_METRICS_FINAL/reports_stream/Drift_Log.csv")
ALT_LOG_PATH = os.path.join(PROJECT_ROOT, "baocao/CD_AHAL_FINAL_FULL_METRICS_FINAL/reports_stream/Drift_Log_CD_AHAL.csv")
OUTPUT_IMG_PATH = os.path.join(PROJECT_ROOT, "baocao/IMAGES")

if not os.path.exists(OUTPUT_IMG_PATH):
    os.makedirs(OUTPUT_IMG_PATH)

def main():
    print(f"📂 Đang tìm log tại: {LOG_PATH}")
    
    final_log_path = LOG_PATH
    if not os.path.exists(LOG_PATH):
        if os.path.exists(ALT_LOG_PATH):
            print(f"⚠️ Không thấy Drift_Log.csv, dùng file thay thế: {ALT_LOG_PATH}")
            final_log_path = ALT_LOG_PATH
        else:
            print(f"❌ Lỗi nghiêm trọng: Không tìm thấy file log nào!")
            print(f"   Đường dẫn gốc kiểm tra: {PROJECT_ROOT}")
            return

    try:
        df = pd.read_csv(final_log_path)
        print(f"✅ Đã load {len(df)} sự kiện Drift.")
    except Exception as e:
        print(f"❌ Lỗi đọc file CSV: {e}")
        return

    # Tính toán thống kê
    avg_lat = df['Latency(ms)'].mean()
    med_lat = df['Latency(ms)'].median()
    max_lat = df['Latency(ms)'].max()
    min_lat = df['Latency(ms)'].min()

    # Style
    sns.set_style("whitegrid")
    
    # ---------------------------------------------------------
    # VẼ BIỂU ĐỒ KÉP (2 Subplots)
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2, 1]})
    
    # --- 1. TIMELINE (Biến động theo thời gian) ---
    ax1.plot(df['Batch'], df['Latency(ms)'], color='#1f77b4', linestyle='-', linewidth=1.5, alpha=0.5)
    scatter = ax1.scatter(df['Batch'], df['Latency(ms)'], c=df['Latency(ms)'], cmap='viridis', s=80, edgecolor='k', zorder=10)
    
    ax1.axhline(y=avg_lat, color='#d62728', linestyle='--', linewidth=2, label=f'Avg: {avg_lat:.0f} ms')
    
    ax1.set_title("Adaptation Latency over Time (Timeline)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Drift Event (Batch Index)", fontsize=12)
    ax1.set_ylabel("Retraining Latency (ms)", fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.5)
    plt.colorbar(scatter, ax=ax1, label='Latency (ms)')
    
    # --- 2. DISTRIBUTION (Phân phối tần suất) ---
    sns.histplot(df['Latency(ms)'], kde=True, bins=15, color='#2ca02c', edgecolor='black', ax=ax2, alpha=0.7)
    
    ax2.axvline(x=avg_lat, color='#d62728', linestyle='--', linewidth=2, label=f'Mean: {avg_lat:.0f}ms')
    ax2.axvline(x=med_lat, color='blue', linestyle=':', linewidth=2, label=f'Median: {med_lat:.0f}ms')
    
    ax2.set_title("Latency Distribution", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Latency (ms)", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.legend()
    
    # Lưu ảnh
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_IMG_PATH, "CD_AHAL_Latency_DeepDive.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Đã lưu biểu đồ thành công: {save_path}")
    
    # ---------------------------------------------------------
    # XUẤT SỐ LIỆU ĐỂ COPY VÀO BÁO CÁO
    # ---------------------------------------------------------
    print("\n" + "="*40)
    print("📊 THỐNG KÊ ĐỘ TRỄ (Dùng cho Báo cáo)")
    print("="*40)
    print(f"• Tổng số lần cập nhật: {len(df)}")
    print(f"• Thời gian trung bình: {avg_lat:.2f} ms")
    print(f"• Thời gian trung vị:   {med_lat:.2f} ms")
    print(f"• Nhanh nhất (Min):     {min_lat:.2f} ms")
    print(f"• Chậm nhất (Max):      {max_lat:.2f} ms")
    print("="*40)

if __name__ == "__main__":
    main()