import streamlit as st
home = st.Page("Pages/Dashboard.py", title="Dashboard", icon="📊")
ml = st.Page("Pages/ml.py", title="Sistema predictivo", icon="📈")
cluster = st.Page("Pages/cluster.py", title="Clustering", icon="🔍")

pg = st.navigation([home, ml, cluster])
pg.run()