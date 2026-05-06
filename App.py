import streamlit as st
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from io import BytesIO
import warnings

warnings.filterwarnings("ignore")

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Research Analytics Engine",
    layout="wide"
)

# =========================
# FUTURISTIC UI CSS
# =========================

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
    border-right: 1px solid rgba(0,212,255,0.2);
}

.block-container {
    padding-top: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

h1, h2, h3 {
    color: white;
}

.stSelectbox label,
.stTextInput label,
.stNumberInput label {
    color: #7dd3fc !important;
}

.stButton button {
    background: linear-gradient(90deg,#00d4ff,#7c3aed);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 10px 18px;
    font-weight: 600;
}

.stButton button:hover {
    opacity: 0.9;
}

div[data-baseweb="select"] {
    background: #0b1220;
    border-radius: 10px;
}

input {
    background: #0b1220 !important;
    color: white !important;
}

[data-testid="stDataFrame"] {
    background: #081122;
    border-radius: 16px;
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)

# =========================
# CONFIG
# =========================

FILE_MAP = {
    "Fine-grained (0.5 threshold, ~9000)": "clusters_dt0.5_with_year_counts.json",
    "Moderate (1.0 threshold, ~3000)": "clusters_dt1_with_year_counts.json",
    "Broad (1.5 threshold, ~1600)": "clusters_dt1.5_with_year_counts.json"
}

# =========================
# LOAD DATA
# =========================

@st.cache_data
def load_all_clusters():
    data = {}

    for label, file in FILE_MAP.items():
        with open(file, "r", encoding="utf-8") as f:
            clusters = json.load(f)
            data[label] = clusters

    return data

cluster_data = load_all_clusters()

# =========================
# HEADER
# =========================

st.title("📈 Research Analytics Engine")
st.caption("Advanced Material Science Keyword Trend Analysis")

# =========================
# SIDEBAR
# =========================

st.sidebar.title("⚡ Dashboard Controls")

selected_granularity = st.sidebar.selectbox(
    "Select specificity level:",
    list(FILE_MAP.keys())
)

allow_multi = st.sidebar.checkbox("Allow multi-select")

selected_data = cluster_data[selected_granularity]

# =========================
# KEYWORD SEARCH
# =========================

st.subheader("🔍 Keyword Search")

query = st.text_input("Type keyword to search:")

filtered_keywords = sorted([
    kw for kw in selected_data
    if query.lower() in kw.lower()
])

selected_keywords = []

if allow_multi:
    selected_keywords = st.multiselect(
        "Select keywords:",
        filtered_keywords
    )
elif filtered_keywords:
    selected_keywords = [
        st.selectbox(
            "Select keyword:",
            filtered_keywords
        )
    ]

# =========================
# PLOT FUNCTION
# =========================

def plot_keywords(keywords, data):

    years = [y for y in range(2011, 2025) if y != 2019]

    df = pd.DataFrame(index=years)

    for kw in keywords:
        year_counts = data.get(kw, {})

        df[kw] = [
            year_counts.get(str(y), 0)
            for y in years
        ]

    st.markdown("### 📈 Trend Analysis")

    st.line_chart(df)

    st.markdown("### 📊 Comparative Distribution")

    st.bar_chart(df)

    return df

# =========================
# DISPLAY PLOTS
# =========================

if selected_keywords:

    st.subheader("📊 Trend over Years")

    df = plot_keywords(
        selected_keywords,
        selected_data
    )

    st.info(
        "ℹ️ Data for 2019 is unavailable and excluded from charts."
    )

    fig, ax = plt.subplots()

    df.plot(ax=ax)

    ax.set_title("Keyword Trends")
    ax.set_xlabel("Year")
    ax.set_ylabel("Count")

    buf = BytesIO()

    fig.savefig(buf, format="png")

    st.download_button(
        "📥 Download Plot as PNG",
        data=buf.getvalue(),
        file_name="keyword_trends.png",
        mime="image/png"
    )

# =========================
# TOP TECHNOLOGIES
# =========================

st.markdown("---")

st.subheader("🏆 Top Technologies by Year")

top_n = st.number_input(
    "Enter number of technologies:",
    min_value=1,
    max_value=100,
    value=10
)

selected_year = st.selectbox(
    "Select a year:",
    list(range(2011, 2025))
)

if st.button("Show Top Technologies"):

    keyword_counts = {
        kw: int(
            selected_data.get(kw, {}).get(str(selected_year), 0)
        )
        for kw in selected_data
    }

    sorted_keywords = sorted(
        keyword_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top_keywords = sorted_keywords[:top_n]

    if top_keywords:

        top_df = pd.DataFrame(
            top_keywords,
            columns=["Keyword", f"Count in {selected_year}"]
        )

        st.dataframe(top_df)

        fig, ax = plt.subplots()

        ax.barh(
            [kw for kw, _ in reversed(top_keywords)],
            [count for _, count in reversed(top_keywords)]
        )

        ax.set_xlabel("Count")
        ax.set_title(
            f"Top {top_n} Technologies in {selected_year}"
        )

        st.pyplot(fig)

    else:
        st.warning("No data available for selected year.")
