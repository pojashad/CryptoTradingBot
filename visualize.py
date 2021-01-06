import streamlit as st
import pandas as pd
st.cache(suppress_st_warning=True)
df = pd.read_csv("trading.csv")
cols = ['Close', 'Open']
st_ms = st.multiselect("Columns", df.columns.tolist(), default=cols)
st.line_chart(df[st_ms].head(1000))
