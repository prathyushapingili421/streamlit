import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt

# -----------------------
# Load Model + Metadata
# -----------------------
@st.cache_resource
def load_resources():
    try:
        with open("customer_kmeans_model_prathyusha.pkl", "rb") as f:
            model = pickle.load(f)

        with open("customer_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        return model, metadata
    
    except Exception as e:
        st.error(f"Error loading model or metadata: {e}")
        return None, None


model, metadata = load_resources()

# If loading failed, stop
if model is None:
    st.stop()

# Extract metadata
feature_names = metadata["feature_names"]
cluster_centers = metadata["cluster_centers"]
sim_data = metadata["simulated_data"]

# -----------------------
# Streamlit App UI
# -----------------------
st.title("💼 Customer Segmentation App (K-Means Clustering)")
st.markdown("Predict which customer group a user belongs to.")

st.sidebar.header("Customer Input")

# -----------------------
# Sidebar Input Sliders
# -----------------------
income = st.sidebar.slider(
    feature_names[0], 
    min_value=20, max_value=150, value=60
)

spending_score = st.sidebar.slider(
    feature_names[1], 
    min_value=1, max_value=100, value=50
)

# -----------------------
# Prediction Logic
# -----------------------
user_input = np.array([[income, spending_score]])

cluster_id = model.predict(user_input)[0]

# Display metric
st.metric(
    label="Predicted Cluster ID",
    value=int(cluster_id)
)

# -----------------------
# Scatter Plot
# -----------------------
st.subheader("Customer Segmentation Plot")

fig, ax = plt.subplots()

# Plot the simulated data
ax.scatter(
    sim_data["Annual Income (k$)"],
    sim_data["Spending Score"],
    c="lightgray",
    label="Simulated Customers"
)

# Plot cluster centers
ax.scatter(
    cluster_centers[:, 0],
    cluster_centers[:, 1],
    c="red",
    s=200,
    marker="X",
    label="Cluster Centers"
)

# Highlight user point
ax.scatter(
    income,
    spending_score,
    c="blue",
    s=150,
    label="Your Customer",
    edgecolors="black"
)

ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score")
ax.legend()

st.pyplot(fig)
