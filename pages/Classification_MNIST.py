import streamlit as st
from buoi4.Classification import Classification

st.title("🔢 Classification MNIST")
st.write("Ứng dụng Classification MNIST đang chạy...")

# Gọi hàm Classification từ module
Classification()
