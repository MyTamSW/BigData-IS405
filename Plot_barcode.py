import pandas as pd
import numpy as np
from ripser import ripser
import matplotlib.pyplot as plt
from pathlib import Path

def custom_plot_persistence_diagram(csv_file, title="Persistence Diagram"):
    # Đọc dữ liệu
    data = pd.read_csv(csv_file, header=None).values

    # Ripser
    diagrams = ripser(data)['dgms']
    h0 = diagrams[0]
    h1 = diagrams[1]

    # Xử lý NaN/Inf: thay thế bằng giá trị max giả định (giống trong mã bạn gửi)
    h0 = np.nan_to_num(h0, nan=1.41421, posinf=1.41421)
    h1 = np.nan_to_num(h1, nan=1.41421, posinf=1.41421)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.title(title)
    
    # H0
    if len(h0) > 0:
        plt.scatter(h0[:, 0], h0[:, 1], c='blue', label='H0', alpha=0.6)
    # H1
    if len(h1) > 0:
        plt.scatter(h1[:, 0], h1[:, 1], c='orange', label='H1', alpha=0.6)

    # Đường chéo
    max_val = max(np.max(h0[:,1]), np.max(h1[:,1]), 1.0)
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)

    plt.xlabel("Birth")
    plt.ylabel("Death")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

# === THAY FILE Ở ĐÂY ===
level = "L5"
label = "Churn"   # hoặc "No_Churn"
index = 1

desktop = Path.home() / "Desktop"
csv_file = desktop / f"{label}_Landmarks" / level / f"{label}_{index}.csv"

# === VẼ BIỂU ĐỒ ===
custom_plot_persistence_diagram(csv_file, title=f"{label} - {level}")
