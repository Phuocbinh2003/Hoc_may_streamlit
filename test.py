import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml

def explain_pca():
    st.markdown(r"""
    ## 🧠 Principal Component Analysis (PCA)
    PCA là một phương pháp giảm chiều dữ liệu bằng cách tìm các thành phần chính (principal components), tức là các trục mới sao cho dữ liệu được trải rộng nhất theo các hướng này.

    ### 🔹 **Các bước thực hiện PCA:**
    1. Chuẩn hóa dữ liệu để có giá trị trung bình bằng 0.
    2. Tính ma trận hiệp phương sai (Covariance Matrix):
       $$
       C = \frac{1}{n} X^T X
       $$
    3. Tính giá trị riêng ($\lambda$) và vector riêng ($v$) của ma trận hiệp phương sai:
       $$
       C v = \lambda v
       $$
    4. Chọn $k$ vector riêng tương ứng với $k$ giá trị riêng lớn nhất để tạo không gian mới.
    5. Biểu diễn dữ liệu trong không gian mới:
       $$
       X_{new} = X W
       $$
       với $W$ là ma trận chứa các vector riêng.

    ### ✅ **Ưu điểm của PCA**
    - Giảm chiều nhanh chóng.
    - Bảo toàn thông tin quan trọng trong dữ liệu.
    - Loại bỏ nhiễu trong dữ liệu.

    ### ❌ **Nhược điểm của PCA**
    - Không giữ được cấu trúc phi tuyến tính của dữ liệu.
    - Các thành phần chính không dễ giải thích về mặt ý nghĩa.
    """)


def explain_tsne():
    st.markdown(r"""
    ## 🌌 t-Distributed Stochastic Neighbor Embedding (t-SNE)
    t-SNE là một phương pháp giảm chiều mạnh mẽ, giúp hiển thị dữ liệu đa chiều trên mặt phẳng 2D hoặc không gian 3D bằng cách bảo toàn mối quan hệ giữa các điểm gần nhau.

    ### 🔹 **Nguyên lý hoạt động của t-SNE:**
    1. **Tính xác suất điểm gần nhau trong không gian gốc:**  
       Với mỗi điểm $x_i$, ta định nghĩa xác suất có điều kiện giữa $x_i$ và $x_j$ như sau:
       $$
       p_{j|i} = \frac{\exp(-\| x_i - x_j \|^2 / 2\sigma^2)}{\sum_{k \neq i} \exp(-\| x_i - x_k \|^2 / 2\sigma^2)}
       $$
       Trong đó, $\sigma$ là tham số ảnh hưởng đến mức độ phân bố của điểm xung quanh.

    2. **Tính xác suất trong không gian giảm chiều (2D/3D):**  
       Trong không gian giảm chiều, ta sử dụng phân phối t-Student với một mức độ tự do:
       $$
       q_{j|i} = \frac{(1 + \| y_i - y_j \|^2)^{-1}}{\sum_{k \neq i} (1 + \| y_i - y_k \|^2)^{-1}}
       $$

    3. **Tối ưu hóa khoảng cách giữa $p_{j|i}$ và $q_{j|i}$:**  
       t-SNE cố gắng làm cho $p_{j|i}$ trong không gian gốc gần bằng $q_{j|i}$ trong không gian mới, bằng cách giảm **hàm mất mát Kullback-Leibler (KL divergence):**
       $$
       KL(P||Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}
       $$

    ### ✅ **Ưu điểm của t-SNE**
    - Hiển thị cụm dữ liệu tốt hơn PCA.
    - Bảo toàn mối quan hệ phi tuyến tính.

    ### ❌ **Nhược điểm của t-SNE**
    - Chạy chậm hơn PCA.
    - Nhạy cảm với các tham số như perplexity.
    """)
def thi_nghiem():
    
    st.title("📉 Giảm chiều dữ liệu MNIST với PCA & t-SNE")

    # Load dữ liệu
    Xmt = np.load("buoi4/X.npy")
    ymt = np.load("buoi4/y.npy")
    X = Xmt.reshape(Xmt.shape[0], -1) 
    y = ymt.reshape(-1) 






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
    
    
def pca_tsne():
        
    tab1, tab2, tab3 = st.tabs(["📘 Lý thuyết PCA", "📘 Lý thuyết t-NSE", "📘 Data"] )

    with tab1:
        explain_pca()

    with tab2:
        explain_tsne()
    
    with tab3:
        thi_nghiem()



if __name__ == "__main__":
    pca_tsne()