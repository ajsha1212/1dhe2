# ==========================================
# Model Lead
# train_model.py
# Qëllimi: Trajnimi i modelit SVD dhe ruajtja e tij
# ==========================================
print("SCRIPT STARTED")
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import joblib
import os
import sys

# ------------------------------------------
# HAPI 1: Gjej rrugën e saktë të dataset-it
# Script ndodhet në: model_lead/src/train_model.py
# Dataset ndodhet në: Data_Preprocessing_Lead_Project/data/processed/ratings_matrix.csv
# ------------------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(
    script_dir,
    "..",
    "..",
    "Data_Preprocessing_Lead_Project",
    "data",
    "processed",
    "ratings_matrix.csv"
)

# ------------------------------------------
# HAPI 2: Folder ku do ruhet modeli
# ------------------------------------------

model_folder = os.path.join(script_dir, "..", "models")
os.makedirs(model_folder, exist_ok=True)
model_path = os.path.join(model_folder, "svd_model.pkl")

# ------------------------------------------
# HAPI 3: Ngarkojme dataset-in
# ------------------------------------------

print("Duke ngarkuar dataset-in...")
print("Rruga:", data_path)

if not os.path.exists(data_path):
    print("❌ Gabim: Dataset nuk u gjet.")
    sys.exit(1)

data = pd.read_csv(data_path, index_col=0)
print("✅ Dataset u ngarkua me sukses!")

# ------------------------------------------
# HAPI 4: Përgatitja e të dhënave 
# ------------------------------------------

data = data.apply(pd.to_numeric, errors="coerce").fillna(0.0)
data_matrix = data.values.astype(np.float64)

print("Forma e të dhënave:", data_matrix.shape)

# ------------------------------------------
# HAPI 5: Trajnimi i modelit SVD
# ------------------------------------------

n_components = min(50, min(data_matrix.shape) - 1)

svd = TruncatedSVD(n_components=n_components, random_state=42)
svd.fit(data_matrix)

print(f"✅ Modeli SVD u trajnu me sukses! (n_components={n_components})")

# ------------------------------------------
# HAPI 6: Ruajtja e modelit
# ------------------------------------------

joblib.dump(svd, model_path)
print("✅ Modeli u ruajt në:", model_path)

# ------------------------------------------
# HAPI 7: Test i shpejtë
# ------------------------------------------

reduced_data = svd.transform(data_matrix)
print("Forma e të dhënave pas SVD:", reduced_data.shape)
print("5 rreshtat e parë:\n", reduced_data[:5])
