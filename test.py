import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml

# @st.cache_data
# def load_mnist():
#     """Tải dữ liệu MNIST từ OpenML"""
#     mnist = fetch_openml('mnist_784', version=1, as_frame=False)
#     X, y = mnist.data, mnist.target
#     return X, y.astype(int)

st.title("📉 Giảm chiều dữ liệu MNIST với PCA & t-SNE")

# Load dữ liệu
X = np.load("buoi4/X.npy")
y = np.load("buoi4/y.npy")

st.write(X[0])



# Tùy chọn thuật toán
method = st.radio("Chọn phương pháp giảm chiều", ["PCA", "t-SNE"])
n_components = st.slider("Số chiều giảm xuống", 2, 3, 2)

# Giới hạn số mẫu để tăng tốc (có thể chỉnh sửa)
num_samples = 5000
X_subset, y_subset = X[:num_samples], y[:num_samples]

# Nút chạy thuật toán
if st.button("🚀 Chạy giảm chiều"):
    with st.spinner("Đang xử lý..."):
        if method == "PCA":
            reducer = PCA(n_components=n_components)
        else:
            reducer = TSNE(n_components=n_components, perplexity=min(30, num_samples - 1), random_state=42)

        X_reduced = reducer.fit_transform(X_subset)

        # Hiển thị kết quả
        if n_components == 2:
            fig = px.scatter(x=X_reduced[:, 0], y=X_reduced[:, 1], color=y_subset.astype(str),
                             title=f"{method} giảm chiều xuống {n_components}D",
                             labels={'x': "Thành phần 1", 'y': "Thành phần 2"})
        else:  # Biểu đồ 3D
            fig = px.scatter_3d(x=X_reduced[:, 0], y=X_reduced[:, 1], z=X_reduced[:, 2],
                                color=y_subset.astype(str),
                                title=f"{method} giảm chiều xuống {n_components}D",
                                labels={'x': "Thành phần 1", 'y': "Thành phần 2", 'z': "Thành phần 3"})

        st.plotly_chart(fig)

st.success("Hoàn thành!")
