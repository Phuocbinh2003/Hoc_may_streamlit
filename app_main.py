import streamlit as st
from buoi2.tien_su_ly_du_lieu import main 
from buoi4.Classification import Classification
from buoi5.Clustering_Algorithms import ClusteringAlgorithms

# Lấy query parameter từ URL
query_params = st.query_params
app_option = query_params.get("app", "home")  # Mặc định là 'home'

# Nếu đổi ứng dụng, reset session để giải phóng tài nguyên
if "last_app" not in st.session_state:
    st.session_state.last_app = app_option

if st.session_state.last_app != app_option:
    st.session_state.clear()
    st.session_state.last_app = app_option
    st.experimental_rerun()  # Reload lại trang để xóa bộ nhớ cũ

# Chạy ứng dụng tương ứng
if app_option == "linear_regression":
    main()
elif app_option == "classification":
    Classification()
elif app_option == "clustering":
    ClusteringAlgorithms()
else:
    st.write("🔍 Vui lòng chọn một ứng dụng từ thanh điều hướng.")
    st.write("📝 Các lựa chọn:")
    st.write("- [Linear Regression](?app=linear_regression)")
    st.write("- [Classification MNIST](?app=classification)")
    st.write("- [Clustering Algorithms](?app=clustering)")
