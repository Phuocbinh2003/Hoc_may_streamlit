import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml

def explain_pca():
    st.markdown("## 🧠 Hiểu PCA một cách đơn giản")

    st.markdown("""
    **PCA (Phân tích thành phần chính)** là một phương pháp giúp giảm số chiều của dữ liệu mà vẫn giữ được thông tin quan trọng nhất.  
    Hãy tưởng tượng bạn có một tập dữ liệu nhiều chiều (nhiều cột), nhưng bạn muốn biểu diễn nó trong không gian 2D hoặc 3D để dễ hiểu hơn. PCA giúp bạn làm điều đó!  

    ### 🔹 **Ví dụ trực quan**:
    Hãy tưởng tượng bạn có một tập dữ liệu gồm nhiều điểm phân bố theo một đường chéo trong không gian 2D:
    """)

   
    np.random.seed(42)
    x = np.random.rand(100) * 10  
    y = x * 0.8 + np.random.randn(100) * 2  
    X = np.column_stack((x, y))

    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], color="blue", alpha=0.5, label="Dữ liệu ban đầu")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.legend()
    st.pyplot(fig)

    st.markdown("""
    Dữ liệu này có sự phân tán rõ ràng theo một hướng chính. PCA sẽ tìm ra hướng đó để biểu diễn dữ liệu một cách tối ưu.

    ### 🔹 **Các bước thực hiện PCA một cách dễ hiểu**:
    1️⃣ **Tìm điểm trung tâm (mean vector)**  
       Tính giá trị trung bình của từng cột (từng chiều dữ liệu).  
       
    2️⃣ **Dịch chuyển dữ liệu về gốc tọa độ**  
       Trừ mỗi điểm dữ liệu đi giá trị trung bình để tập trung dữ liệu quanh gốc.  

    3️⃣ **Tính ma trận hiệp phương sai**  
       Hiểu đơn giản, ma trận này đo mức độ các biến thay đổi cùng nhau.  

    4️⃣ **Tìm các hướng quan trọng nhất**  
       - Tính các trị riêng (eigenvalues) và vector riêng (eigenvectors).  
       - Chúng cho ta biết đâu là hướng quan trọng nhất của dữ liệu.  

    5️⃣ **Chiếu dữ liệu lên không gian mới**  
       - Chọn một số hướng chính (principal components).  
       - Biểu diễn dữ liệu theo các trục này thay vì trục gốc.  

    ### 🔹 **Trực quan hóa quá trình PCA**
    Dưới đây là minh họa cách PCA tìm ra trục quan trọng nhất của dữ liệu:
    """)

    # PCA thủ công
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], color="blue", alpha=0.5, label="Dữ liệu ban đầu")
    origin = np.mean(X, axis=0)

    for i in range(2):
        ax.arrow(origin[0], origin[1], 
                 eigenvectors[0, i] * 3, eigenvectors[1, i] * 3, 
                 head_width=0.3, head_length=0.3, color="red", label=f"Trục {i+1}")

    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.legend()
    st.pyplot(fig)

    st.markdown("""
    **🔹 Kết quả:**  
    - Trục đỏ là hướng mà PCA tìm ra.  
    - Nếu chọn 1 trục chính, ta có thể chiếu dữ liệu lên nó để giảm chiều.  
      
    Nhờ đó, chúng ta có thể biểu diễn dữ liệu một cách gọn gàng hơn mà không mất quá nhiều thông tin!  
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