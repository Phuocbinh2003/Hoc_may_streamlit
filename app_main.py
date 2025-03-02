import streamlit as st
import subprocess

# Lấy query parameter từ URL
query_params = st.query_params
app_option = query_params.get("app", "home")

app_mapping = {
    "linear_regression": "buoi2.tien_su_ly_du_lieu.py",
    "classification": "buoi4.Classification.py",
    "clustering": "buoi5.Clustering_Algorithms.py"
}

if app_option in app_mapping:
    subprocess.Popen(["streamlit", "run", app_mapping[app_option]])
else:
    st.write("🔍 Vui lòng chọn một ứng dụng từ thanh điều hướng.")
    st.write("📝 Các lựa chọn:")
    st.write("- [Linear Regression](?app=linear_regression)")
    st.write("- [Classification MNIST](?app=classification)")
    st.write("- [Clustering Algorithms](?app=clustering)")
