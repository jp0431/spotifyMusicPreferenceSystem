import streamlit as sl
import pandas as pd
import  joblib
import datetime as dt
import plotly.express as px

sl.set_page_config(
    page_title="AI Music Predictor",
    page_icon="🤖",
    layout="wide"
)



songs_df = pd.read_csv("Data/song_ml.csv")
songs_df["has_features"] = songs_df["energy"].notna()
songs_df = songs_df[songs_df["has_features"] == True]
modelo = joblib.load("Models/predict_skip.pkl")
scaler = joblib.load("Models/scaler_skip.pkl")
features = [
    
    "acousticness",
    "danceability",
    "energy",
    "tempo",
    "valence",
    "hour",
    "month",
    "cluster",
    "loudness",           
    "speechiness",        
    "instrumentalness",  
    "liveness"       

]
sl.title("🤖 AI Music Predictor")


dict_cluster = {0.0: "Oscuro Intenso",
                1.0: "Acústico Contemplativo", 
                2.0: "Energético Alegre",
                3.0: "Íntimo y Melancólico"}
sl.markdown(
    """
Predeice si escucharé una canción completa, basado en:
- Comportamiento de escucha
- Spotify audio features
- Habitos de escucha temporales
"""
)
sl.divider()
sel_song = sl.selectbox(
    "🎵 Selecciona una canción: ",
    songs_df["query"].unique()
)
if sl.button("Analiza canción"):
    song_df = songs_df[songs_df["query"] == sel_song].drop_duplicates(subset="query")[features]
    song_df["hour"] = dt.datetime.now().hour #Obtenemos la hora actual
    song_df["month"] = dt.datetime.now().month #Obtenemos el mes actual
    cluster = int(song_df["cluster"].iloc[0])
    Scaler_song_df= scaler.transform(song_df)
    prediction = modelo.predict(Scaler_song_df)
    skip_proba = modelo.predict_proba(Scaler_song_df)[0][1]
    listen_prob = 1-skip_proba
    col1, col2 = sl.columns([2, 1])
    with col1:

        sl.subheader("❤️ Probabilidad de escucha")
        sl.metric(
            label="Probability of full listen",
            value=f"{listen_prob:.0%}"
        )
        sl.progress(float(listen_prob))
        if listen_prob > 0.7:
            sl.success("🔥 Alta probabilidad de que disfrute esta canción")
        elif listen_prob > 0.4:
            sl.warning("🎧 Predicción neutral")
        else:
            sl.error("⏭️ Alta probabilidad de skip")
    with col2:

        sl.subheader("🧠 Cluster Musical")
        sl.metric(
            "Cluster",
            dict_cluster.get(cluster, "Desconocido")
        )  
    sl.divider()
    sl.subheader("🎚️ Audio Features")

    feature_df = pd.DataFrame({
        "Feature": [
            "Danceability",
            "Energy",
            "Valence",
            "Acousticness",
            "Tempo"
        ],
        "Value": [
            song_df["danceability"].iloc[0],
            song_df["energy"].iloc[0],
            song_df["valence"].iloc[0],
            song_df["acousticness"].iloc[0],
            song_df["tempo"].iloc[0],
        ]
    })

    fig = px.bar(
        feature_df,
        x="Feature",
        y="Value",
        template="plotly_dark",
        title="Spotify Audio Features"
    )

    sl.plotly_chart(fig, use_container_width=True)
    sl.subheader("🤖 ¿Porque de esta predicción?")

    insights = []

    if song_df["energy"].iloc[0] > 0.7:
        insights.append("⚡ Las canciones más energicas son las que más escucho")

    if song_df["acousticness"].iloc[0] > 0.7:
        insights.append("🎻 Canciones acusticas coinciden con mis habitos")

    if dt.datetime.now().hour >= 22:
        insights.append("🌙 Escucha nocturna detectada")

    if len(insights) == 0:
        insights.append("🎧 Comportamiento Mixto detectado")

    for insight in insights:
        sl.write(insight)

