# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 12:23:39 2021

@author: msagming
"""
import pandas as pd
import numpy as np
import gudhi
from sklearn import preprocessing
import matplotlib.pyplot as pls 
from pathlib import Path

# === Load and normalize dataset ===
dataset = pd.read_csv('Churn_tuned_dataset.csv', header=None)
dataset = dataset.values
min_max_scaler = preprocessing.MinMaxScaler()
dataset_scaled = min_max_scaler.fit_transform(dataset)
norm_dataset = pd.DataFrame(dataset_scaled)

# === Split churn / non-churn ===
label_col = norm_dataset.shape[1] - 1
churn_dataset = norm_dataset[norm_dataset[label_col] == 1]
no_churn_dataset = norm_dataset[norm_dataset[label_col] == 0].sample(n=3672, random_state=42)

witnesses_churn = np.array(churn_dataset)
witnesses_no_churn = np.array(no_churn_dataset)

# === Config landmark sizes ===
landmark_levels = {
    "L5": 184,    # 5% of 3672
    "L10": 367,   # 10%
    "L15": 551    # 15%
}

# === Create folders on Desktop ===
desktop = Path.home() / "Desktop"
for level in landmark_levels:
    (desktop / f"Churn_Landmarks/{level}").mkdir(parents=True, exist_ok=True)
    (desktop / f"No_Churn_Landmarks/{level}").mkdir(parents=True, exist_ok=True)

# === Generate landmarks for all levels ===
for i in range(500):
    for level, size in landmark_levels.items():
        # Generate churn landmark
        landmarks_churn = gudhi.pick_n_random_points(points=witnesses_churn, nb_points=size)
        df_churn = pd.DataFrame(landmarks_churn)
        df_churn.to_csv(desktop / f"Churn_Landmarks/{level}/Churn_{i+1}.csv", index=False, header=False)

        # Generate non-churn landmark
        landmarks_no_churn = gudhi.pick_n_random_points(points=witnesses_no_churn, nb_points=size)
        df_no_churn = pd.DataFrame(landmarks_no_churn)
        df_no_churn.to_csv(desktop / f"No_Churn_Landmarks/{level}/No_Churn_{i+1}.csv", index=False, header=False)

print("Done: Generated L5, L10, L15 landmark sets for churn and non-churn on Desktop.")