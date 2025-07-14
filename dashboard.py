import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
from datetime import datetime

st.set_page_config(page_title="NYC Taxi Dashboard", layout="wide")

# ---- Load Data ----
@st.cache_data
def load_data():
    df = pd.read_parquet("data/sample_trips.parquet")
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])
    df['trip_duration_min'] = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds() / 60
    df['hour'] = df['pickup_datetime'].dt.hour
    return df

df = load_data()

# ---- Sidebar Filters ----
st.sidebar.title("🛠️ Filters")
hour_range = st.sidebar.slider("Select Pickup Hour", 0, 23, (6, 20))
df_filtered = df[(df['hour'] >= hour_range[0]) & (df['hour'] <= hour_range[1])]

# ---- Header ----
st.title("🚕 NYC Taxi Trip Dashboard")
st.markdown("Explore taxi trip data with interactive charts, maps and insights.")

# ---- Tabs ----
tab1, tab2, tab3 = st.tabs(["📈 Trip Analytics", "🗺️ Pickup Map", "📊 Hourly Demand"])

# ---- Tab 1: Trip Analytics ----
with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Trips", f"{len(df_filtered):,}")
    col2.metric("Avg Duration (min)", f"{df_filtered['trip_duration_min'].mean():.2f}")
    col3.metric("Max Duration (min)", f"{df_filtered['trip_duration_min'].max():.2f}")

    st.subheader("📊 Trip Duration Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df_filtered['trip_duration_min'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

# ---- Tab 2: Folium Map ----
with tab2:
    st.subheader("🗺️ Pickup Locations")
    m = folium.Map(location=[40.75, -73.97], zoom_start=12)
    for _, row in df_filtered.sample(min(100, len(df_filtered))).iterrows():
        folium.CircleMarker(
            location=[row['pickup_latitude'], row['pickup_longitude']],
            radius=3,
            fill=True,
            fill_opacity=0.7,
            color='blue'
        ).add_to(m)
    st_data = st_folium(m, width=700)

# ---- Tab 3: Hourly Demand ----
with tab3:
    st.subheader("🕐 Pickup Demand by Hour")
    hourly_counts = df_filtered['hour'].value_counts().sort_index()
    fig2, ax2 = plt.subplots()
    sns.barplot(x=hourly_counts.index, y=hourly_counts.values, palette="Blues_d", ax=ax2)
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Number of Pickups")
    st.pyplot(fig2)

# ---- Custom Styling ----
st.markdown("""
    <style>
        .main { background-color: #f9f9f9; }
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        .stTabs [data-baseweb="tab"] { font-size: 18px; }
    </style>
""", unsafe_allow_html=True)
