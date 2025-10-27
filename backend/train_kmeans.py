import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import joblib

# Paths
CLEANED_FILE = Path("data/processed/cleaned_features.csv")
MODEL_DIR = Path("backend/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "kmeans_model.joblib"
SCALER_FILE = MODEL_DIR / "scaler.joblib"
MAPPING_FILE = MODEL_DIR / "label_mapping.joblib"

# 1. Load cleaned features
df = pd.read_csv(CLEANED_FILE)
print("Shape:", df.shape)

# 2. Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df)

# 3. Train K-Means
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_scaled)

# 4. Metrics
inertia = kmeans.inertia_
labels = kmeans.labels_
sil_score = silhouette_score(X_scaled, labels)
cluster_counts = pd.Series(labels).value_counts()

print(f"Inertia: {inertia:.4f}")
print(f"Silhouette score: {sil_score:.4f}")
print("Cluster counts:")
print(cluster_counts)
print("Centroids (scaled space):")
print(kmeans.cluster_centers_)

# 5. Map clusters to lifespan buckets
label_mapping = {
    0: "CLUSTER_3_8_10_years",
    1: "CLUSTER_2_5_8_years",
    2: "CLUSTER_1_lt_5_years"
}

# 6. Save model, scaler, and mapping
joblib.dump(kmeans, MODEL_FILE)
joblib.dump(scaler, SCALER_FILE)
joblib.dump(label_mapping, MAPPING_FILE)

print(f"Saved model to {MODEL_FILE}")
print(f"Saved scaler to {SCALER_FILE}")
print(f"Saved label mapping to {MAPPING_FILE}")
