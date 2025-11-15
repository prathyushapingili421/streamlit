import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import pickle

# -------------------------------
# 1. SIMULATE CUSTOMER DATA
# -------------------------------
np.random.seed(42)  # for same results every time

# Create 200 customers
annual_income = np.random.randint(20, 150, 200)   # income between $20k–$150k
spending_score = np.random.randint(1, 100, 200)   # spending score 1–100

# Combine into a DataFrame
data = pd.DataFrame({
    "Annual Income (k$)": annual_income,
    "Spending Score": spending_score
})

# Features for training
X = data[["Annual Income (k$)", "Spending Score"]].values

# -------------------------------
# 2. TRAIN K-MEANS MODEL
# -------------------------------
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)

# -------------------------------
# 3. SAVE TRAINED MODEL
# -------------------------------
model_filename = "customer_kmeans_model_yourname.pkl"
with open(model_filename, "wb") as f:
    pickle.dump(kmeans, f)

# -------------------------------
# 4. SAVE METADATA
# -------------------------------
metadata = {
    "feature_names": ["Annual Income (k$)", "Spending Score"],
    "cluster_centers": kmeans.cluster_centers_,
    "simulated_data": data  # Save original data for scatter plot
}

with open("customer_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("✔ Model and metadata saved successfully!")
