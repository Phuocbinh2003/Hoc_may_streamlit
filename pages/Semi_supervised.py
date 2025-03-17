import streamlit as st
from buoi8.Semi_supervised import Semi_supervised

if "last_page" in st.session_state and st.session_state.last_page != "Semi_supervised":
    st.session_state.clear()  # Xóa toàn bộ session

st.session_state.last_page = "Semi_supervised" 



# Gọi hàm Classification từ module
Semi_supervised()
