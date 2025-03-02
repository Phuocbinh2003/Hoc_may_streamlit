import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml

@st.cache_data
def load_mnist():
    """Tải dữ liệu MNIST từ OpenML"""
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target
    return X, y.astype(int)

st.title("📉 Giảm chiều dữ liệu MNIST với PCA & t-SNE")

# Load dữ liệu
X, y = load_mnist()

# Tùy chọn thuật toán
method = st.radio("Chọn phương pháp giảm chiều", ["PCA", "t-SNE"])
n_components = st.slider("Số chiều giảm xuống", 2, 3, 2)

# Nút chạy thuật toán
if st.button("🚀 Chạy giảm chiều"):
    with st.spinner("Đang xử lý..."):
        if method == "PCA":
            reducer = PCA(n_components=n_components)
        else:
            reducer = TSNE(n_components=n_components, perplexity=30, random_state=42)

        X_reduced = reducer.fit_transform(X[:5000])  # Chỉ dùng 5000 ảnh để tăng tốc

        # Hiển thị kết quả
        fig, ax = plt.subplots(figsize=(6, 6))
        scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y[:5000], cmap="jet", alpha=0.5)
        ax.set_title(f"{method} giảm chiều xuống {n_components}D")
        st.pyplot(fig)

st.success("Hoàn thành!")
