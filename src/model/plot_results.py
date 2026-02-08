import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- CẤU HÌNH ĐƯỜNG DẪN CHÍNH XÁC ---
# Dựa trên đường dẫn bạn cung cấp: D:\1.Cong_Viec\NguyencuuKhoaHoc\CD_AHAL\src\model
# Chúng ta cần trỏ về D:\1.Cong_Viec\NguyencuuKhoaHoc\CD_AHAL\baocao

# Lấy thư mục hiện tại của file code
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Đi ngược lên 2 cấp để về root (CD_AHAL)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))

RESULT_PATH = os.path.join(PROJECT_ROOT, "baocao/BASELINES_RESULT")
CD_AHAL_LOG_PATH = os.path.join(PROJECT_ROOT, "baocao/CD_AHAL_FINAL_FULL_METRICS_FINAL/reports_stream/Drift_Log.csv")
OUTPUT_IMG_PATH = os.path.join(PROJECT_ROOT, "baocao/IMAGES")

if not os.path.exists(OUTPUT_IMG_PATH):
    os.makedirs(OUTPUT_IMG_PATH)

print(f"📂 Đang tìm dữ liệu tại: {RESULT_PATH}")

def main():
    print(">>> ĐANG XỬ LÝ DỮ LIỆU VÀ VẼ BIỂU ĐỒ...")

    # 1. LOAD DỮ LIỆU BASELINES
    try:
        df_adwin = pd.read_csv(os.path.join(RESULT_PATH, "Baseline_ADWIN.csv"))
        df_arf = pd.read_csv(os.path.join(RESULT_PATH, "Baseline_ARF.csv"))
        df_odl = pd.read_csv(os.path.join(RESULT_PATH, "Baseline_ODL.csv"))
        df_periodic = pd.read_csv(os.path.join(RESULT_PATH, "Baseline_PERIODIC.csv"))
        
        df_cdahal = pd.read_csv(CD_AHAL_LOG_PATH)
        print("✅ Đã load thành công tất cả file CSV.")
    except Exception as e:
        print(f"❌ Lỗi đọc file: {e}")
        print("👉 Hãy kiểm tra xem bạn đã chạy xong 3 file baseline chưa?")
        return

    # ---------------------------------------------------------
    # BIỂU ĐỒ 1: ACCURACY TRAJECTORY
    # ---------------------------------------------------------
    plt.figure(figsize=(14, 7))
    window = 5
    
    # Plot Baselines
    plt.plot(df_arf['Batch'], df_arf['Accuracy'].rolling(window).mean(), 
             label='Adaptive Random Forest (SOTA)', color='#2ca02c', linewidth=2, linestyle='--')
    plt.plot(df_odl['Batch'], df_odl['Accuracy'].rolling(window).mean(), 
             label='Online Deep Learning (ODL)', color='#ff7f0e', linewidth=2, alpha=0.8)
    plt.plot(df_periodic['Batch'], df_periodic['Accuracy'].rolling(window).mean(), 
             label='Periodic Retraining', color='#9467bd', linewidth=1.5, alpha=0.7)
    plt.plot(df_adwin['Batch'], df_adwin['Accuracy'].rolling(window).mean(), 
             label='CNN-BiGRU + ADWIN', color='#7f7f7f', linewidth=1.5, linestyle=':', alpha=0.6)

    # Plot CD-AHAL Recovery Points
    plt.scatter(df_cdahal['Batch'], df_cdahal['Recov_Acc'], 
                color='#d62728', s=100, marker='*', label='CD-AHAL Recovery', zorder=10)
    
    # Plot CD-AHAL Trend Line (Interpolation)
    # Tạo đường xu hướng từ các điểm drift để so sánh trực quan
    x_drift = df_cdahal['Batch'].values
    y_drift = df_cdahal['Recov_Acc'].values
    # Thêm điểm đầu và cuối nếu cần để vẽ hết biểu đồ
    if x_drift[0] != 0:
        x_drift = np.insert(x_drift, 0, 0)
        y_drift = np.insert(y_drift, 0, y_drift[0])
    if x_drift[-1] < 247:
        x_drift = np.append(x_drift, 247)
        y_drift = np.append(y_drift, y_drift[-1])
        
    plt.plot(x_drift, y_drift, color='#d62728', linewidth=2.5, label='CD-AHAL (Trend)', alpha=0.9)

    plt.title("Comparative Adaptation Performance under Concept Drift", fontsize=16, fontweight='bold')
    plt.xlabel("Streaming Batches", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.ylim(0.6, 1.0)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12, loc='lower right')
    
    save_path1 = os.path.join(OUTPUT_IMG_PATH, "Comparison_Accuracy_Trajectory.png")
    plt.savefig(save_path1, dpi=300, bbox_inches='tight')
    print(f"✅ Đã lưu biểu đồ 1: {save_path1}")

    # ---------------------------------------------------------
    # BIỂU ĐỒ 2: LATENCY & COST
    # ---------------------------------------------------------
    metrics = []
    
    # CD-AHAL
    avg_lat_cdahal = df_cdahal['Latency(ms)'].mean()
    updates_cdahal = len(df_cdahal)
    metrics.append(['CD-AHAL', avg_lat_cdahal, updates_cdahal])
    
    # ODL
    avg_lat_odl = df_odl[df_odl['Train_Time_ms'] > 0]['Train_Time_ms'].mean()
    updates_odl = len(df_odl[df_odl['Train_Time_ms'] > 0])
    metrics.append(['Online DL', avg_lat_odl, updates_odl])
    
    # ARF
    avg_lat_arf = df_arf['Total_Time_ms'].mean() 
    updates_arf = len(df_arf)
    metrics.append(['Adaptive RF', avg_lat_arf, updates_arf])
    
    # Periodic
    avg_lat_per = df_periodic[df_periodic['Train_Time_ms'] > 0]['Train_Time_ms'].mean()
    updates_per = len(df_periodic[df_periodic['Train_Time_ms'] > 0])
    metrics.append(['Periodic', avg_lat_per, updates_per])

    df_metrics = pd.DataFrame(metrics, columns=['Model', 'Latency (ms)', 'Update Count'])

    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#9467bd']
    sns.barplot(data=df_metrics, x='Model', y='Update Count', ax=ax1, palette=colors, alpha=0.6)
    ax1.set_ylabel("Number of Updates (Resource Cost)", fontsize=14, color='black')
    ax1.bar_label(ax1.containers[0], padding=3)
    
    ax2 = ax1.twinx()
    sns.lineplot(data=df_metrics, x='Model', y='Latency (ms)', ax=ax2, color='black', marker='o', markersize=10, linewidth=2, sort=False)
    ax2.set_ylabel("Avg. Latency per Update (ms)", fontsize=14, color='black')
    ax2.set_ylim(0, 1500)
    
    plt.title("Computational Cost Analysis", fontsize=16, fontweight='bold')
    
    save_path2 = os.path.join(OUTPUT_IMG_PATH, "Comparison_Latency_Cost.png")
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    print(f"✅ Đã lưu biểu đồ 2: {save_path2}")

if __name__ == "__main__":
    main()