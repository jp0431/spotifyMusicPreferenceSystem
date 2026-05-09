import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

df = pd.read_csv("Data/song_ml.csv")
kmeans = joblib.load("Models/kmeans.pkl")
scaler = joblib.load("Models/scaler_cl.pkl")
pca = joblib.load("Models/pca.pkl")
cluster_features = [
  "acousticness", "danceability", "energy",   "loudness",   "valence"]
df = df.dropna(subset=cluster_features)
X_scaled = scaler.transform(df[cluster_features])
X_pca = pca.transform(X_scaled)
df["cluster"] = kmeans.predict(X_scaled)
pca_df = pd.DataFrame({
    "PCA1": X_pca[:, 0],
    "PCA2": X_pca[:, 1],
    "cluster": df["cluster"].astype(str),
    "track": df["trackName"],
    "artist": df["artistName"]
})

cluster_counts = (
    df["cluster"]
    .value_counts(normalize=True)
    .reset_index()
)
st.title("🎵 Clusters Musicales")

st.markdown("""
Descubrimiento de tipos de música usando KMeans clustering.
""")

col1, col2 = st.columns(2)

with col1:
    st.metric("Clusters", "4")

with col2:
    st.metric("Best Silhouette Score", "0.26")


fig = px.scatter(
    pca_df,
    x="PCA1",
    y="PCA2",
    color="cluster",
    hover_data=["track", "artist"],
    template="plotly_dark",
    title="Distribución PCA de canciones",
    opacity=0.8,
    color_discrete_sequence=['#337ec4', '#50c778', '#ffffc2', '#bf5455']
)

fig.update_layout(
    height=700,
    title_font_size=24,
    legend_title="Cluster"
)

st.plotly_chart(fig, use_container_width=True)
st.divider()
st.subheader("🎧 Tipos de clusters")

col1, col2 = st.columns(2)
datos = df.groupby("cluster")[["artistName", "trackName"]].value_counts().reset_index()
canciones = []
for i in range (0,4):
    datas = datos[datos["cluster"] == i]
    song = datas.loc[datas['count'].idxmax(), ['trackName', 'artistName']].iloc[0]
    print(f"Cluster: {i} Canción: {song}")
    canciones.append(song)




with col1:

    st.info(f"""
    🎻 Cluster 0  
    Oscuro e intenso    
    Ejemplo: {canciones[0]}
    """)

    st.success(f"""
    ⚡ Cluster 1  
    Acústico contemplativo  
    Ejemplo: {canciones[1]}
    """)

with col2:

    st.warning(f"""
    🌑 Cluster 2  
    Energético y alegre  
    Ejemplo: {canciones[2]}  
    """)

    st.error(f"""
    🎤 Cluster 3  
    Íntimo y melancólico  
    Ejemplo: {canciones[3]}  
    """)

st.subheader("""
Distribución de clusters:   
""")
pie = px.pie(
    cluster_counts,
    names="cluster",
    values="proportion",
    hole=0.5,
    color_discrete_sequence=['#337ec4', '#50c778', '#ffffc2', '#bf5455']
)
st.plotly_chart(pie , use_container_width=True)

