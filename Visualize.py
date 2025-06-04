# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from ripser import ripser
# from pathlib import Path

# # Định nghĩa đường dẫn gốc
# base_dir = Path.home() / "Desktop"
# groups = ["Churn_Landmarks", "No_Churn_Landmarks"]
# landmark_levels = ["L5", "L10", "L15"]

# MAX_FILES = 10  # Số file tối đa xử lý trong mỗi folder

# # === Hàm vẽ barcode stacked ===
# def plot_barcode_stacked(dgms, title, save_path=None):
#     colors = ['black', 'blue', 'red']
#     labels = [r'$H_0$', r'$H_1$', r'$H_2$']

#     fig, axs = plt.subplots(len(dgms), 1, figsize=(8, 8), sharex=True)

#     for i, (ax, dgm) in enumerate(zip(axs, dgms)):
#         # Nếu dgm rỗng thì bỏ qua
#         if dgm.shape[0] == 0:
#             continue
#         for j, interval in enumerate(dgm):
#             birth, death = interval
#             # Nếu death = inf, vẽ 1 đoạn dài cho biểu diễn
#             if np.isinf(death):
#                 death = birth + 1.0  # Tùy chỉnh đoạn dài cho dễ nhìn
#             ax.hlines(y=j, xmin=birth, xmax=death, color=colors[i], lw=2)
#         ax.set_ylabel(labels[i], rotation=0, fontsize=14, labelpad=20)
#         ax.set_ylim(-1, len(dgm))
#         ax.grid(False)

#     axs[-1].set_xlabel("Filtration Value", fontsize=12)
#     fig.suptitle(title, fontsize=14)
#     plt.tight_layout(rect=[0, 0, 1, 0.96])  # Để title không bị đè

#     if save_path:
#         plt.savefig(save_path)
#     else:
#         plt.show()

#     plt.close(fig)

# # === Xử lý từng thư mục ===
# for group in groups:
#     for level in landmark_levels:
#         folder_path = base_dir / group / level
#         label = f"{group}_{level}"
#         print(f"⏳ Đang xử lý: {folder_path}")

#         if not folder_path.exists():
#             print(f"❌ Không tìm thấy thư mục: {folder_path}")
#             continue

#         files = sorted([f for f in os.listdir(folder_path) if f.endswith(".csv")])[:MAX_FILES]

#         for file_name in files:
#             file_path = folder_path / file_name

#             try:
#                 data = pd.read_csv(file_path, header=None).values
#                 # Kiểm tra data có ít nhất 2 cột để tính ripser
#                 if data.shape[1] < 2:
#                     print(f"⚠️ File {file_name} có ít hơn 2 chiều, bỏ qua.")
#                     continue

#                 dgms = ripser(data, maxdim=2)['dgms']

#                 out_dir = Path("output_visuals") / label
#                 out_dir.mkdir(parents=True, exist_ok=True)
#                 save_file = out_dir / f"{file_name.replace('.csv', '')}.png"

#                 plot_barcode_stacked(dgms, title=file_name, save_path=save_file)
#                 print(f"✅ Đã lưu: {save_file}")

#             except Exception as e:
#                 print(f"⚠️ Lỗi khi xử lý {file_name}: {e}")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ripser import ripser
from pathlib import Path

# === Cấu hình ===
base_dir = Path.home() / "Desktop"
groups = ["Churn_Landmarks", "No_Churn_Landmarks"]
landmark_levels = ["L5", "L10", "L15"]
MAX_FILES = 500  # Số file tối đa xử lý mỗi thư mục

# === Hàm trích xuất đặc trưng từ diagram ===
def summarize_dgms(dgms):
    summary = []
    for dgm in dgms:
        if len(dgm) == 0:
            summary.append((0, 0.0))  # (số lượng, trung bình độ dài)
        else:
            lengths = [death - birth for birth, death in dgm if not np.isinf(death)]
            avg_len = np.mean(lengths) if lengths else 0.0
            summary.append((len(dgm), avg_len))
    return summary  # Trả về [(count_H0, avg_len_H0), (count_H1, avg_len_H1), (count_H2, avg_len_H2)]

# === Thu thập tất cả đặc trưng ===
summary_stats = []

for group in groups:
    for level in landmark_levels:
        folder_path = base_dir / group / level
        label = f"{group}_{level}"
        print(f"⏳ Đang xử lý thư mục: {folder_path}")

        if not folder_path.exists():
            print(f"❌ Không tìm thấy: {folder_path}")
            continue

        files = sorted([f for f in os.listdir(folder_path) if f.endswith(".csv")])[:MAX_FILES]

        for file_name in files:
            file_path = folder_path / file_name
            try:
                data = pd.read_csv(file_path, header=None).values
                if data.shape[1] < 2:
                    continue  # Không đủ chiều để tính ripser

                dgms = ripser(data, maxdim=2)['dgms']
                summary = summarize_dgms(dgms)

                for dim, (count, avg_len) in enumerate(summary):
                    summary_stats.append({
                        "Group": group.replace("_Landmarks", ""),
                        "Landmark_Level": level,
                        "Dimension": f"H{dim}",
                        "Count": count,
                        "Avg_Length": avg_len
                    })

            except Exception as e:
                print(f"⚠️ Lỗi xử lý {file_name}: {e}")
                continue

# === Chuyển sang DataFrame ===
df = pd.DataFrame(summary_stats)

# === Tạo thư mục lưu hình ===
output_img_dir = Path("tda_visual_outputs")
output_img_dir.mkdir(exist_ok=True)

# === Vẽ từng hình riêng biệt ===
sns.set(style="whitegrid")

for level in landmark_levels:
    for group in ["Churn", "No_Churn"]:
        # Lọc dữ liệu cho từng nhóm
        df_subset = df[(df["Landmark_Level"] == level) & (df["Group"] == group)]
        if df_subset.empty:
            print(f"⚠️ Không có dữ liệu cho {group} - {level}")
            continue

        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df_subset, x="Dimension", y="Avg_Length", palette="Set2")

        plt.title(f"{group} - {level}: Avg Persistence Length")
        plt.xlabel("TDA Dimension")
        plt.ylabel("Avg Persistence Length")
        plt.tight_layout()

        # Lưu ảnh
        img_path = output_img_dir / f"{group}_{level}_AvgLength.png"
        plt.savefig(img_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✅ Đã lưu biểu đồ: {img_path.resolve()}")
