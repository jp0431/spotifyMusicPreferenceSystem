import streamlit as sl
import pandas as pd
import numpy as np
import plotly.express as px
import ipywidgets as widgets
from IPython.display import display
import plotly.io as pio

# Funciones 
def incializa_metricas(data): #Obtenemos las metricas más importantes de las canciones de spotify
    
    #Canciones repetidas
    top_repes = data[data["skipped"]== False][['trackName', "artistName"]].value_counts().reset_index().head(10)
    #Canciones más escuchadas
    top_canciones = data.groupby("trackName")['msPlayed'].sum().sort_values(ascending=False).reset_index().head(5)
    #Artitstas escuchadas
    top_artistas = data.groupby("artistName")['msPlayed'].sum().sort_values(ascending=False).reset_index().head(5)
    #Total horas escuchadas
    horas_escuchadas = data["HoursPlayed"].sum()
    #total Canciones escuchcadas
    n_canciones = len(data['trackName'])
    #total artistas escuchados
    n_artits = data['artistName'].nunique()
    #Total canciones skipeadas
    skips = len(data[data["skipped"] == True])
    #% canciones skipeadas
    percen_skips = (skips * 100 /n_canciones)
    artists_per_month = data.groupby("month_entropy")["artistName"].nunique()
    avg_artist = artists_per_month.mean()
    artist_counts = data["artistName"].value_counts(normalize=True)
    #La entropia nos permite calcular la diversidad de escucha que junto el número de artistas por mes obtenemos la exploración musical
    entropy = -np.sum(artist_counts * np.log(artist_counts))
    score = 0.7 * avg_artist + 0.3 * entropy
    return top_canciones, top_artistas,top_repes, horas_escuchadas, n_canciones, n_artits, skips, percen_skips, score,avg_artist
def crear_graficos(top_canciones,top_artistas, top_repes, data): #Esta función crea los diferentes graficos para el dashboard.
    top_canciones['minPlayed'] = top_canciones['msPlayed'] / 60000
    fig_songs = px.bar(top_canciones,x='trackName',y='minPlayed', title="Top Canciones escuchadas", labels={"trackName": "Canción", "minPlayed":"Minutos"})
    
    top_artistas['minPlayed'] = top_artistas['msPlayed'] / 60000
    fig_repes = px.bar(top_repes,x='trackName',y='count', title="Top canciones más repetidas", labels={"trackName": "Canción", "count":"Veces Repetida"})
    fig_artist = px.bar(top_artistas,x='artistName',y='minPlayed', title="Top artitas escuchados", labels={"artistName": "Autor", "minPlayed":"Minutos"})
    dias= data.groupby("day")["HoursPlayed"].sum().reset_index()
    #dict_dia_sem = {'Monday':'lunes', 'Tuesday':'martes', 'Wednesday':'miercoles', "Thursday":'jueves', "Friday":'viernes', "Saturday":'sabado', "Sunday":'domingo'}
    #data['day'] = data['day'].map(dict_dia_sem)
    heatmap = data.pivot_table(
        index="day",
        columns="hour",
        values="HoursPlayed",
        aggfunc="sum"
    ).fillna(0)
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    #days_order = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"]

    heatmap = heatmap.reindex(days_order)


    fig_heatmap = px.imshow(
        heatmap,
        aspect="auto",
            color_continuous_scale=[   "#0b0f2b",
        "#3b0f70",
        "#8c2981",
        "#de4968",
        "#fe9f6d",
        "#fcfdbf"],
        labels=dict(x="Hora del día", y="Día", color="Horas"),
        template="plotly_dark",

    )
    fig_heatmap.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),

    )
    return fig_songs, fig_artist, fig_repes,fig_heatmap
sl.set_page_config(
    page_title="Spotify ML Dashboard",
    layout="wide"
)   
# CONFIG
sl.set_page_config("SPOTIFY DASHBOARD")
sl.title("SPOTIFY DASHBOARD Q1 2026")
# Cargar datos
data1 = pd.read_json("Data/StreamingHistory_music_0.json")
data2 = pd.read_json("Data/StreamingHistory_music_1.json")
dfs = [data1, data2] #Convertimos el tipo de datos de la columna endTime
for df in dfs:
    df["endTime"] = pd.to_datetime(df["endTime"])
data1 = data1[data1["endTime"] >= "2026-01-01"] #Seleccionamos los datos a partir de 2026
data2 = data2[data2["endTime"] < "2026-04-01"]#Seleccionamos los datos hasta abril de 2026
data = pd.concat([data1, data2])
#Creamos las columnas de horas Escuchadas, minutos escuchados, la hora, el día y si se ha skipeado la canción
data["HoursPlayed"] = data["msPlayed"] / (1000*60*60)
data["MinutesPlayed"] = data["msPlayed"] /  60000
data["hour"] = data["endTime"].dt.hour
data["month"] = data["endTime"].dt.month
data["day"] = data["endTime"].dt.day_name()
data["skipped"] = data["msPlayed"] < 30000
data["month_entropy"] = data["endTime"].dt.to_period("M")



# Añadimos los 2 filtros, mes y artista
sl.sidebar.header("Filtros")
month = sl.sidebar.selectbox("Selecciona mes", ["Todos"]+sorted(data["month"].unique()))
# artista = sl.sidebar.selectbox("Selecciona el artisa", ["Todos"]+sorted(data["artistName"].unique()))
# # Ajustamos el comportamiento del filtro
# df = data.copy()
# if month != "Todos":
#     df = df[df["month"] == month]
# if artista != "Todos":
#     df = df[df["artistName"] == artista]
# if df.empty:
#     sl.warning("No hay datos para esta selección")
#     sl.stop()
df_temp = data.copy()
if month != "Todos":
    df_temp = df_temp[df_temp["month"] == month]

artistas = sl.sidebar.multiselect(
    "Selecciona el artista",
    sorted(df_temp["artistName"].unique())
)
df = df_temp.copy()

if artistas:
    df = df[df["artistName"].isin(artistas)]

# Generamos las metricas cada vez que se aplique un filtro
top_canciones, top_artistas,top_repes, horas_escuchadas, n_canciones, n_artits, skips, percen_skips, score, avg_artist = incializa_metricas(df)
sl.write(
    f"He escuchado {horas_escuchadas:.0f} horas de música "
    f"y descubierto {n_artits} artistas diferentes."
)
c1, c2,c3,c4 = sl.columns(4)
c1.metric("🎧 Horas", f"{horas_escuchadas:.1f}")
c2.metric("🎵 Canciones escuchadas", f"{n_canciones}")
c3.metric("🎤 Artitstas", f"{n_artits}")
c4.metric("⏭️ % Skips: ",f"{percen_skips:.2f}")

#Graficos

fig_songs, fig_artist, fig_repes,fig_heatmap = crear_graficos(top_canciones,top_artistas, top_repes, df)

sl.markdown("## 🕒 ¿Cuándo escucho música?")
dict_dia_sem = {'Monday':'lunes', 'Tuesday':'martes', 'Wednesday':'miercoles', "Thursday":'jueves', "Friday":'viernes', "Saturday":'sabado', "Sunday":'domingo'}
sl.plotly_chart(fig_heatmap)
horas = df.groupby("hour")["HoursPlayed"].sum().reset_index()
horas = horas.sort_values("HoursPlayed", ascending= False)["hour"].iloc[0]
hora_dia = df.groupby(["hour", "day"])['HoursPlayed'].sum().reset_index()#.sort_values("HoursPlayed", ascending=False).iloc[0]
hora_dia['day'] =hora_dia['day'].map(dict_dia_sem)
hora_dia = hora_dia.sort_values("HoursPlayed", ascending=False).iloc[0]
sl.markdown(f"🎯 Suelo escuchar música a las **{int(horas)}:00**. Pero el día de la semana que más escucho música es el **{hora_dia.iloc[1]}** a las **{int(hora_dia.iloc[0])}:00**")

sl.markdown("## 🧠 Mi perfil musical")

# Exploración musical
text = ""
if score < 20:
    text = "Baja"
elif score  >= 20 and score < 81: 
    text = "media"
elif score > 80 and score <151:
    text = "alta"
elif score > 150:
    text = "Muy alta"

col1, col2 = sl.columns(2)
col1.metric("🔍 Exploración", text)
col2.metric("🎧 Engagement", f"{100 - percen_skips:.0f}%")

#Sección artistas
top_artista =  top_artistas.iloc[0]["artistName"]
top_artista_count = top_artistas.iloc[0]["msPlayed"] / df["msPlayed"].sum() * 100
if len(artistas) >1: #Si hay un artista seleccionado no se mostrará quien es el artista favorito ni el tiempo escuchado
    sl.markdown("## Artistas")
    sl.write(f"Mi artista favorito es **{top_artista}**", f"Representando un **{top_artista_count:.2f}%** del tiempo")
    sl.plotly_chart(fig_artist)
sl.markdown("## 🎶 Tus canciones favoritas")
p1, p2,= sl.columns(2)
with p1:
    sl.plotly_chart(fig_songs)
with p2:
    sl.plotly_chart(fig_repes)

sl.markdown("## ⏭️ Cómo escucho música")

sl.write(
    f"Salto aproximadamente el **{percen_skips:.1f}%** de las canciones, "
    f"lo que indica un comportamiento "
    f"{'explorador' if percen_skips > 30 else 'selectivo'}."
)
sl.markdown("## 🎯 Conclusión")
horario = ""
if horas > 6 and horas <=11:  horario = "la mañana"
elif horas > 11 and horas < 14: horario ="el mediodia"
elif horas >=14 and horas < 20: horario = "la tarde"
elif horas >= 20 and horas < 2: horario = "la noche"
elif horas >= 2 and horas <= 6: horario = "la madrugada" 

sl.write(
    f"Soy un oyente con un perfil **{text}**, "
    f"que escucha principalmente por **{horario}** y "
    f"explora una gran variedad de artistas, {avg_artist:.0f} por mes."
)

sl.info("Usa el menú lateral para navegar entre páginas.")