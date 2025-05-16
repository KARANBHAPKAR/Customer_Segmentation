import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Set up Streamlit page
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("🧠 Customer Segmentation using K-Means Clustering")

# --- Load Data ---
@st.cache_data
def load_default_data():
    return pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mall_customers.csv")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data uploaded successfully!")
else:
    df = load_default_data()
    st.info("ℹ️ Using default Mall Customers dataset.")

# Show raw data
if st.checkbox("Show raw data"):
    st.write(df)

# --- Column Mapping ---
st.sidebar.header("🛠️ Column Mapping (Optional)")
columns = df.columns.tolist()

# Optional mapping for Gender column
gender_col = st.sidebar.selectbox("Select Gender Column (if exists)", ["None"] + columns)
if gender_col != "None":
    df[gender_col] = df[gender_col].map({'Male': 0, 'Female': 1, 'M': 0, 'F': 1})
    df[gender_col] = df[gender_col].fillna(0)  # default to 0 if mapping fails

# --- Feature Selection ---
df_clean = df.select_dtypes(include=[np.number]).dropna()
numeric_cols = df_clean.columns.tolist()

st.sidebar.header("🔧 Select Features")
features = st.sidebar.multiselect("Choose features for clustering:", numeric_cols, default=numeric_cols)

if len(features) < 2:
    st.warning("⚠️ Select at least two features for clustering.")
    st.stop()

X = df_clean[features]
X_scaled = StandardScaler().fit_transform(X)

# --- KMeans Clustering ---
k = st.sidebar.slider("Select number of clusters (K)", 2, 10, 5)
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df_result = df_clean.copy()
df_result["Cluster"] = clusters

# --- PCA for Visualization ---
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)
df_result["PCA1"] = components[:, 0]
df_result["PCA2"] = components[:, 1]

# --- Cluster Visualization ---
st.subheader("📊 Cluster Visualization (PCA Projection)")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df_result, x="PCA1", y="PCA2", hue="Cluster", palette="Set2", s=100, ax=ax)
ax.set_title("Customer Segments (2D PCA View)")
st.pyplot(fig)

# --- Cluster Summary ---
st.subheader("📋 Cluster Summary")
summary = df_result.groupby("Cluster")[features].mean()
summary["Count"] = df_result["Cluster"].value_counts()
st.dataframe(summary)

# --- Silhouette Score ---
score = silhouette_score(X_scaled, clusters)
st.sidebar.metric("Silhouette Score", f"{score:.2f}")

# --- Download Button ---
csv = df_result.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Clustered Data", data=csv, file_name="clustered_customers.csv", mime="text/csv")