import streamlit as st
from buoi7.Neural_Network import Neural_Network
if "last_page" in st.session_state and st.session_state.last_page != "Neural_Network":
    st.session_state.clear()  # Xóa toàn bộ session

st.session_state.last_page = "Neural_Network" 

st.title("🔍 TNeural Network")


# Gọi hàm ClusteringAlgorithms từ module
Neural_Network()
    