# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 12:44:37 2021

@author: Sagming Marcel
"""

#Importing libraries
import numpy as np
import pandas as pd
from ripser import ripser
from pathlib import Path
import random

def Ripser_Code():
    desktop = Path.home() / "Desktop"

    churn_dirs = {
        0: ("No_Churn_Landmarks", "No_Churn"),
        1: ("Churn_Landmarks", "Churn")
    }

    landmark_levels = ["L5", "L10", "L15"]

    # Thư mục lưu file kết quả là thư mục hiện hành (nơi chạy script)
    output_dir = Path.cwd()

    for level in landmark_levels:
        output_filename = output_dir / f"Dataset_Statistics_{level}.txt"
        # Xóa file cũ nếu có
        if output_filename.exists():
            output_filename.unlink()

        for j in [0, 1]:
            dir_name, file_prefix = churn_dirs[j]
            base_path = desktop / dir_name / level  # Đường dẫn đúng theo cấu trúc bạn cung cấp

            for i in range(500):
                filename = base_path / f"{file_prefix}_{i+1}.csv"
                if not filename.exists():
                    print(f"Không tìm thấy file: {filename}")
                    continue

                landmark_set = pd.read_csv(filename, header=None)

                points = ripser(landmark_set)['dgms']

                points_h0 = np.array(points[0])
                points_h1 = np.array(points[1])

                points_h0_0 = points_h0[:,0]
                points_h0_1 = points_h0[:,1]
                points_h1_0 = points_h1[:,0]
                points_h1_1 = points_h1[:,1]

                points_h0_1[~np.isfinite(points_h0_1)] = 1.41421
                points_h1_0[~np.isfinite(points_h1_0)] = 1.41421
                points_h1_1[~np.isfinite(points_h1_1)] = 1.41421

                length_0 = abs(points_h0_1 - points_h0_0)
                y_max_0 = np.max(points_h0_1)
                ymlength_0 = y_max_0 - points_h0_1

                a11 = np.mean(points_h0_0); a12 = np.mean(points_h0_1); a13 = np.mean(length_0); a14 = np.mean(ymlength_0)
                a21 = np.median(points_h0_0); a22 = np.median(points_h0_1); a23 = np.median(length_0); a24 = np.median(ymlength_0)
                a31 = np.std(points_h0_0); a32 = np.std(points_h0_1); a33 = np.std(length_0); a34 = np.std(ymlength_0)

                length_1 = abs(points_h1_1 - points_h1_0)
                y_max_1 = np.max(points_h1_1)
                ymlength_1 = y_max_1 - points_h1_1

                b11 = np.mean(points_h1_0); b12 = np.mean(points_h1_1); b13 = np.mean(length_1); b14 = np.mean(ymlength_1)
                b21 = np.median(points_h1_0); b22 = np.median(points_h1_1); b23 = np.median(length_1); b24 = np.median(ymlength_1)
                b31 = np.std(points_h1_0); b32 = np.std(points_h1_1); b33 = np.std(length_1); b34 = np.std(ymlength_1)

                M0 = [a11, a12, a13, a14, a21, a22, a23, a24, a31, a32, a33, a34]
                M1 = [b11, b12, b13, b14, b21, b22, b23, b24, b31, b32, b33, b34]

                M = np.concatenate((M0, M1), axis=None)
                M = np.append(M, j)

                Output = ','.join(['%.7f' % num for num in M])

                with open(output_filename, 'a') as file:
                    file.write(f'{Output}\n')

        # Shuffle file sau khi ghi xong mỗi level
        with open(output_filename, 'r') as f:
            lines = f.readlines()
        random.shuffle(lines)
        with open(output_dir / f"Dataset_Statistics_{level}_random.txt", 'w') as f:
            f.writelines(lines)

    print("Hoàn thành xử lý L5, L10, L15 với kết quả lưu tại:", output_dir)

if __name__ == '__main__':
    Ripser_Code()