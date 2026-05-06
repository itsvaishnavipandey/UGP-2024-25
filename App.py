import streamlit as st
st.set_page_config(layout="wide")

st.markdown("""
<style>

.stApp {
    background: #050816;
    color: white;
}

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

[data-testid="stSidebar"] {
    background: #081122;
}

.block-container {
    padding-top: 2rem;
}

.stButton button {
    background: linear-gradient(90deg,#00d4ff,#7c3aed);
    color: white;
    border-radius: 12px;
    border: none;
}

</style>
""", unsafe_allow_html=True)
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from io import BytesIO
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
import warnings
import threading
warnings.filterwarnings("ignore")

# === CONFIG ===
FILE_MAP = {
    "Fine-grained (0.5 threshold, ~9000)": "labeled_clusters_dt0.5.json",
    "Moderate (1.0 threshold, ~3000)": "labeled_clusters_dt1.json",
    "Broad (1.5 threshold, ~1600)": "labeled_clusters_dt1.5.json"
}
DATA_DIR = "./labeled_cluster_results"
EMBED_DIR = "./precomputed_embeddings"

# === Load cluster data once ===
@st.cache_data
def load_all_clusters():
    data = {}
    for label, file in FILE_MAP.items():
        path = os.path.join(DATA_DIR, file)
        with open(path, "r", encoding="utf-8") as f:
            clusters = json.load(f)
            data[label] = clusters
    return data

cluster_data = load_all_clusters()

# === Load model once ===
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

model = load_model()

# === Embed state (dynamic) ===
embedding_state = {
    "keywords": None,
    "embeddings": None,
    "loaded": False,
    "current_granularity": None
}

def load_embeddings_for(granularity_label):
    base = granularity_label.split("(")[0].strip().replace(" ", "_").lower()
    try:
        with open(f"{EMBED_DIR}/{base}_keywords.json", "r") as f:
            embedding_state["keywords"] = json.load(f)
        embedding_state["embeddings"] = np.load(f"{EMBED_DIR}/{base}_embeddings.npy")
        embedding_state["loaded"] = True
        embedding_state["current_granularity"] = granularity_label
    except Exception as e:
        st.error(f"Failed to load embeddings for {granularity_label}: {e}")

# === UI ===
st.title("📈 Materials Keyword Trend Tool")

selected_granularity = st.selectbox("Select specificity level:", list(FILE_MAP.keys()))
selected_data = cluster_data[selected_granularity]

# Load embeddings only when granularity changes
if selected_granularity != embedding_state["current_granularity"]:
    embedding_state["loaded"] = False
    threading.Thread(target=load_embeddings_for, args=(selected_granularity,), daemon=True).start()

allow_multi = st.checkbox("Allow multi-select")
query = st.text_input("Type keyword to search:")
filtered_keywords = sorted([kw for kw in selected_data if query.lower() in kw.lower()])
selected_keywords = []

if allow_multi:
    selected_keywords = st.multiselect("Select keywords:", filtered_keywords)
elif filtered_keywords:
    selected_keywords = [st.selectbox("Select keyword:", filtered_keywords)]

# === Plot ===
def plot_keywords(keywords, data):
    years = [y for y in range(2011, 2025) if y != 2019]  # Remove 2019
    df = pd.DataFrame(index=years)

    for kw in keywords:
        year_counts = data.get(kw, {})
        df[kw] = [year_counts.get(str(y), 0) for y in years]

    st.markdown("📈 **Line Chart**")
    st.line_chart(df)

    st.markdown("📊 **Bar Chart**")
    st.bar_chart(df)

    return df

if selected_keywords:
    st.subheader("📊 Trend over Years")

    # Plotting
    df = plot_keywords(selected_keywords, selected_data)

    # Note about 2019
    st.info("ℹ️ **Note:** Data for the year 2019 is not available and has been excluded from the charts.")

    # Matplotlib download option
    fig, ax = plt.subplots()
    df.plot(ax=ax)
    ax.set_title("Keyword Trends")
    ax.set_xlabel("Year")
    ax.set_ylabel("Count")
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.download_button("📥 Download Plot as PNG", data=buf.getvalue(), file_name="keyword_trends.png", mime="image/png")

# Section separator and top-N tech
st.markdown("---")
st.subheader("🏆 Top N Technologies in a Specific Year")


top_n = st.number_input("Enter the number of top technologies to display:", min_value=1, max_value=100, value=10)
selected_year = st.selectbox("Select a year:", list(range(2011, 2025)))

if st.button("Show Top Technologies"):
    keyword_counts = {
        kw: int(selected_data.get(kw, {}).get(str(selected_year), 0))
        for kw in selected_data
    }
    sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
    top_keywords = sorted_keywords[:top_n]

    if top_keywords:
        top_df = pd.DataFrame(top_keywords, columns=["Keyword", f"Count in {selected_year}"])
        st.dataframe(top_df)

        # Plot bar chart
        fig, ax = plt.subplots()
        ax.barh([kw for kw, _ in reversed(top_keywords)], [count for _, count in reversed(top_keywords)])
        ax.set_xlabel("Count")
        ax.set_title(f"Top {top_n} Technologies in {selected_year}")
        st.pyplot(fig)
    else:
        st.warning("No data available for the selected year.")

# === Semantic Search ===
st.markdown("---")
st.subheader("🔍 Semantic Similar Keywords")

sem_kw = st.text_input("Enter a keyword to find similar ones:", key="sem_kw")

if not embedding_state["loaded"]:
    st.info("⏳ Loading semantic embeddings for selected granularity...")
elif sem_kw:
    emb = model.encode([sem_kw], convert_to_tensor=True)
    similarities = util.cos_sim(emb, torch.tensor(embedding_state["embeddings"]))[0]
    top_k = min(10, len(similarities))
    top_indices = torch.topk(similarities, k=top_k).indices.numpy()
    similar_keywords = [
        embedding_state["keywords"][i]
        for i in top_indices if embedding_state["keywords"][i] != sem_kw
    ]

    st.write("Top similar keywords found:")
    selected_similar_keywords = st.multiselect(
        "Select keywords to add to plot:", similar_keywords, key="semantic_multiselect"
    )

    if selected_similar_keywords:
        st.write("✅ Selected for plotting:", selected_similar_keywords)
        if st.button("Plot selected similar keywords"):
            plot_keywords(selected_similar_keywords, selected_data)
