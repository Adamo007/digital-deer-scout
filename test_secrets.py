import streamlit as st
try:
    token = st.secrets["SENTINELHUB_TOKEN"]
    print("Token:", token[:50] + "...")
except KeyError:
    print("SENTINELHUB_TOKEN not found in secrets")
