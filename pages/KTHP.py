import streamlit as st
from buoicuoi.KTHP import kthp

if "last_page" in st.session_state and st.session_state.last_page != "KTHP":
    st.session_state.clear()  # Xóa toàn bộ session

st.session_state.last_page = "KTHP" 



# Gọi hàm main từ module
kthp()
