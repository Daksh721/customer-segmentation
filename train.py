import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1. Load Raw Dataset
df = pd.read_csv("./dspproj/dataset.csv").drop_duplicates().copy()

# 2. Parse Date Columns
df["SignUpDate"] = pd.to_datetime(df["SignUpDate"], errors="coerce")
df["PurchaseDate"] = pd.to_datetime(df["PurchaseDate"], errors="coerce")
df["LastLogin"] = pd.to_datetime(df["LastLogin"], errors="coerce")

# 3. Clean Text Columns
text_cols = [
    "UserName",
    "Gender",
    "Country",
    "ProductName",
    "Category",
    "ReviewText",
    "DeviceType",
    "ReferralSource",
]
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# 4. Clip Out-of-Range ReviewScore to [0.0, 5.0]
df["ReviewScore"] = (
    pd.to_numeric(df["ReviewScore"], errors="coerce").clip(0.0, 5.0).round(2)
)

# 5. Flag Date Ordering Anomalies & Engineer Features
df["PurchaseBeforeSignUp"] = df["PurchaseDate"] < df["SignUpDate"]
df["EffectiveDiscountRate"] = np.where(
    df["HasDiscountApplied"], df["DiscountRate"], 0.0
).round(4)
df["RecencyDays"] = (df["LastLogin"].max() - df["LastLogin"]).dt.days
df["PurchaseMonth"] = df["PurchaseDate"].dt.to_period("M").astype(str)
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[17, 25, 35, 45, 55, 120],
    labels=["18-25", "26-35", "36-45", "46-55", "56+"],
)

# 6. Prepare Clustering Features
features = ["TotalAmount", "Quantity", "EffectiveDiscountRate", "SessionDuration"]
X = df[features].fillna(df[features].median())

# 7. Evaluate Candidate Cluster Counts (k = 2..7)
X_scaled = StandardScaler().fit_transform(X)
eval_rows = []

for k in range(2, 8):
    km_eval = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_eval = km_eval.fit_predict(X_scaled)
    eval_rows.append(
        {
            "k": k,
            "Inertia": round(km_eval.inertia_, 2),
            "Silhouette": round(
                silhouette_score(X_scaled, labels_eval, sample_size=15000, random_state=42),
                3,
            ),
            "DaviesBouldin": round(davies_bouldin_score(X_scaled, labels_eval), 3),
            "CalinskiHarabasz": round(
                calinski_harabasz_score(X_scaled, labels_eval), 1
            ),
        }
    )

eval_df = pd.DataFrame(eval_rows)
print("Cluster Evaluation:\n", eval_df.to_string(index=False))

# 8. Train Final K-Means Pipeline (k = 4)
best_k = 4
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=best_k, random_state=42, n_init=10)),
    ]
)
df["Cluster"] = pipeline.fit_predict(X)

# 9. Assign Business Segment Labels Dynamically
stats = df.groupby("Cluster")[
    ["TotalAmount", "EffectiveDiscountRate", "SessionDuration"]
].mean()

vip_id = int(stats["TotalAmount"].idxmax())
remaining = stats.drop(index=[vip_id])

discount_id = int(remaining["EffectiveDiscountRate"].idxmax())
remaining = remaining.drop(index=[discount_id])

engaged_id = int(remaining["SessionDuration"].idxmax())
quick_id = int(remaining["SessionDuration"].idxmin())

segment_map = {
    vip_id: "High-Value Bulk Buyers",
    discount_id: "Discount Hunters",
    engaged_id: "Engaged Browsers",
    quick_id: "Quick Casual Shoppers",
}
df["SegmentName"] = df["Cluster"].map(segment_map)

# 10. Save Trained Dataset & Model Artifact
df.to_csv("trained_customer_segments.csv", index=False)

joblib.dump(
    {
        "pipeline": pipeline,
        "features": features,
        "segment_map": segment_map,
        "best_k": best_k,
    },
    "customer_segmentation_model.joblib",
)

print(f"\nSaved trained_customer_segments.csv ({df.shape[0]} rows, {df.shape[1]} cols)")
print("Saved customer_segmentation_model.joblib")
