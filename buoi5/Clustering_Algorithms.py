import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Tải dữ liệu MNIST từ OpenML
def data():
    st.header("MNIST Dataset")
    st.write("""
      **MNIST** là một trong những bộ dữ liệu nổi tiếng và phổ biến nhất trong cộng đồng học máy, 
      đặc biệt là trong các nghiên cứu về nhận diện mẫu và phân loại hình ảnh.
  
      - Bộ dữ liệu bao gồm tổng cộng **70.000 ảnh chữ số viết tay** từ **0** đến **9**, 
        mỗi ảnh có kích thước **28 x 28 pixel**.
      - Chia thành:
        - **Training set**: 60.000 ảnh để huấn luyện.
        - **Test set**: 10.000 ảnh để kiểm tra.
      - Mỗi hình ảnh là một chữ số viết tay, được chuẩn hóa và chuyển thành dạng grayscale (đen trắng).
  
      Dữ liệu này được sử dụng rộng rãi để xây dựng các mô hình nhận diện chữ số.
      """)

    st.subheader("Một số hình ảnh từ MNIST Dataset")
    st.image("buoi4/img3.png", caption="Một số hình ảnh từ MNIST Dataset", use_container_width=True)

    st.subheader("Ứng dụng thực tế của MNIST")
    st.write("""
      Bộ dữ liệu MNIST đã được sử dụng trong nhiều ứng dụng nhận dạng chữ số viết tay, chẳng hạn như:
      - Nhận diện số trên các hoá đơn thanh toán, biên lai cửa hàng.
      - Xử lý chữ số trên các bưu kiện gửi qua bưu điện.
      - Ứng dụng trong các hệ thống nhận diện tài liệu tự động.
    """)

    st.subheader("Ví dụ về các mô hình học máy với MNIST")
    st.write("""
      Các mô hình học máy phổ biến đã được huấn luyện với bộ dữ liệu MNIST bao gồm:
      - **Logistic Regression**
      - **Decision Trees**
      - **K-Nearest Neighbors (KNN)**
      - **Support Vector Machines (SVM)**
      - **Convolutional Neural Networks (CNNs)**
    """)

    st.subheader("Kết quả của một số mô hình trên MNIST ")
    st.write("""
      Để đánh giá hiệu quả của các mô hình học máy với MNIST, người ta thường sử dụng độ chính xác (accuracy) trên tập test:
      
      - **Decision Tree**: 0.8574
      - **SVM (Linear)**: 0.9253
      - **SVM (poly)**: 0.9774
      - **SVM (sigmoid)**: 0.7656
      - **SVM (rbf)**: 0.9823
      
      
      
    """)

def ly_thuyet_K_means():
    
    st.title("📌 K-Means Clustering")

    # 🔹 Giới thiệu về K-Means
    st.markdown("""
    **K-Means Clustering** là thuật toán phân cụm **không giám sát**, giúp chia dữ liệu thành **K cụm** sao cho các điểm trong cùng một cụm có đặc trưng giống nhau nhất.  
    - 📌 **Ý tưởng chính**:  
        1. Chọn ngẫu nhiên **K tâm cụm (centroids)**.  
        2. Gán mỗi điểm dữ liệu vào cụm có tâm gần nhất.  
        3. Cập nhật lại tâm cụm bằng cách lấy trung bình các điểm trong cụm.  
        4. Lặp lại quá trình trên cho đến khi các tâm cụm không thay đổi hoặc số vòng lặp đạt giới hạn.  
    """)

    # 🔹 Công thức khoảng cách Euclidean
    st.latex(r"""
    d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}
    """)
    st.markdown("""
    Trong đó:
    - \( p, q \) là hai điểm trong không gian \( n \) chiều.
    - \( d(p, q) \) là khoảng cách giữa hai điểm.
    """)

    # 🔹 Ưu điểm và Nhược điểm
    st.markdown("### ✅ **Ưu điểm & ❌ Nhược điểm**")
    st.markdown("""
    ✅ **Ưu điểm:**  
    - Đơn giản, dễ hiểu và hiệu quả trên tập dữ liệu lớn.  
    - Chạy nhanh vì thuật toán có độ phức tạp thấp.  

    ❌ **Nhược điểm:**  
    - Cần xác định số cụm \( K \) trước.  
    - Nhạy cảm với giá trị outlier và cách chọn điểm ban đầu.  
    """)

    def euclidean_distance(a, b):
        return np.linalg.norm(a - b, axis=1)

    # Tạo dữ liệu ngẫu nhiên
    def generate_data(n_samples, n_clusters, cluster_std):
        np.random.seed(42)
        X = []
        centers = np.random.uniform(-10, 10, size=(n_clusters, 2))  # Chọn tâm cụm ngẫu nhiên
        for c in centers:
            X.append(c + np.random.randn(n_samples // n_clusters, 2) * cluster_std)
        return np.vstack(X)

    # Hàm khởi tạo tâm cụm ngẫu nhiên
    def initialize_centroids(X, k):
        np.random.seed(None)  # Chọn ngẫu nhiên mỗi lần chạy
        return X[np.random.choice(X.shape[0], k, replace=False)]

    # Hàm gán điểm vào cụm gần nhất
    def assign_clusters(X, centroids):
        labels = np.array([np.argmin(euclidean_distance(x, centroids)) for x in X])
        return labels

    # Hàm cập nhật tâm cụm mới
    def update_centroids(X, labels, k):
        new_centroids = np.array([X[labels == i].mean(axis=0) if len(X[labels == i]) > 0 else np.random.uniform(-10, 10, 2) for i in range(k)])
        return new_centroids

    # Tạo giao diện Streamlit
    st.title("🎯 Minh họa thuật toán K-Means từng bước")

    # Tham số điều chỉnh
    num_samples = st.slider("Số điểm dữ liệu", 50, 500, 200, step=10)
    num_clusters = st.slider("Số cụm (K)", 2, 10, 3)
    cluster_std = st.slider("Độ rời rạc của cụm", 0.5, 3.0, 1.0)

    # Nút Reset để khởi động lại dữ liệu
    if st.button("🔄 Reset"):
        st.session_state.X = generate_data(num_samples, num_clusters, cluster_std)
        st.session_state.centroids = initialize_centroids(st.session_state.X, num_clusters)
        st.session_state.iteration = 0  # Đếm số lần cập nhật
        st.session_state.labels = assign_clusters(st.session_state.X, st.session_state.centroids)

    # Kiểm tra nếu chưa có dữ liệu trong session_state
    if "X" not in st.session_state:
        st.session_state.X = generate_data(num_samples, num_clusters, cluster_std)

    X = st.session_state.X  # Dữ liệu điểm

    # Khởi tạo hoặc cập nhật tâm cụm
    if "centroids" not in st.session_state:
        st.session_state.centroids = initialize_centroids(X, num_clusters)
        st.session_state.iteration = 0
        st.session_state.labels = assign_clusters(X, st.session_state.centroids)

    # Nút cập nhật từng bước
    if st.button("🔄 Cập nhật vị trí tâm cụm"):
        st.session_state.labels = assign_clusters(X, st.session_state.centroids)
        new_centroids = update_centroids(X, st.session_state.labels, num_clusters)
        
        # Kiểm tra xem có thay đổi không, nếu không thì đã hội tụ
        if np.all(new_centroids == st.session_state.centroids):
            st.warning("⚠️ Tâm cụm không thay đổi, thuật toán đã hội tụ!")
        else:
            st.session_state.centroids = new_centroids
            st.session_state.iteration += 1

    # Vẽ biểu đồ
    fig, ax = plt.subplots(figsize=(6, 6))
    labels = st.session_state.labels
    centroids = st.session_state.centroids

    # Vẽ điểm dữ liệu
    for i in range(num_clusters):
        ax.scatter(X[labels == i][:, 0], X[labels == i][:, 1], label=f"Cụm {i}", alpha=0.6, edgecolors="k")

    # Vẽ tâm cụm
    ax.scatter(centroids[:, 0], centroids[:, 1], s=200, c="red", marker="X", label="Tâm cụm")
    ax.set_title(f"Minh họa K-Means (Lần cập nhật: {st.session_state.iteration})")
    ax.legend()

    # Hiển thị biểu đồ
    st.pyplot(fig)


from sklearn.datasets import make_moons, make_blobs
from sklearn.cluster import DBSCAN

def ly_thuyet_DBSCAN():



# Tạo dữ liệu ngẫu nhiên
    def generate_data(n_samples, noise, dataset_type):
        if dataset_type == "Cụm Gauss":
            X, _ = make_blobs(n_samples=n_samples, centers=3, cluster_std=noise, random_state=42)
        else:
            X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=42)
        return X

    # Hàm chạy DBSCAN
    def run_dbscan(X, eps, min_samples):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        return labels

    # Giao diện Streamlit
    st.title("🔍 Minh họa thuật toán DBSCAN")

    # Tùy chỉnh tham số
    num_samples = st.slider("Số điểm dữ liệu", 50, 500, 200, step=10)
    noise = st.slider("Mức nhiễu", 0.05, 1.0, 0.2)
    dataset_type = st.radio("Chọn kiểu dữ liệu", ["Cụm Gauss", "Hai vòng trăng (Moons)"])
    eps = st.slider("Bán kính cụm (eps)", 0.1, 2.0, 0.5, step=0.1)
    min_samples = st.slider("Số điểm tối thiểu để tạo cụm (min_samples)", 2, 20, 5)

    # Nút Reset để tạo lại dữ liệu
    if st.button("🔄 Reset"):
        st.session_state.X = generate_data(num_samples, noise, dataset_type)
        st.session_state.labels = np.full(num_samples, -1)  # Chưa có cụm nào

    # Kiểm tra dữ liệu trong session_state
    if "X" not in st.session_state:
        st.session_state.X = generate_data(num_samples, noise, dataset_type)
        st.session_state.labels = np.full(num_samples, -1)

    X = st.session_state.X

    # Nút chạy DBSCAN
    if st.button("➡️ Chạy DBSCAN"):
        st.session_state.labels = run_dbscan(X, eps, min_samples)

    # Vẽ biểu đồ
    fig, ax = plt.subplots(figsize=(6, 6))
    labels = st.session_state.labels
    unique_labels = set(labels)

    # Màu cho các cụm
    colors = plt.cm.get_cmap("tab10", len(unique_labels))

    for label in unique_labels:
        mask = labels == label
        color = "black" if label == -1 else colors(label)
        ax.scatter(X[mask, 0], X[mask, 1], color=color, label=f"Cụm {label}" if label != -1 else "Nhiễu", edgecolors="k", alpha=0.7)

    ax.set_title(f"Kết quả DBSCAN (eps={eps}, min_samples={min_samples})")
    ax.legend()

    # Hiển thị biểu đồ
    st.pyplot(fig)




# Hàm vẽ biểu đồ
def split_data():
    
    st.title("📌 Chia dữ liệu Train/Test")

    # Đọc dữ liệu
    X = np.load("buoi4/X.npy")
    y = np.load("buoi4/y.npy")
    total_samples = X.shape[0]

    # Thanh kéo chọn số lượng ảnh để train
    num_samples = st.slider("Chọn số lượng ảnh để train:", 1000, total_samples, 10000)

    # Thanh kéo chọn tỷ lệ Train/Test
    test_size = st.slider("Chọn tỷ lệ test:", 0.1, 0.5, 0.2)

    if st.button("✅ Xác nhận & Lưu"):
        # Lấy số lượng ảnh mong muốn
        X_selected, y_selected = X[:num_samples], y[:num_samples]

        # Chia train/test theo tỷ lệ đã chọn
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y_selected, test_size=test_size, random_state=42)

        # Lưu vào session_state để sử dụng sau
        st.session_state["X_train"] = X_train
        st.session_state["y_train"] = y_train
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test

        st.success(f"🔹 Dữ liệu đã được chia: Train ({len(X_train)}), Test ({len(X_test)})")

    # Kiểm tra nếu đã lưu dữ liệu vào session_state
    if "X_train" in st.session_state:
        st.write("📌 Dữ liệu train/test đã sẵn sàng để sử dụng!")
        
def train():
    # 📥 **Tải dữ liệu MNIST**
    if "X_train" in st.session_state:
        X_train = st.session_state["X_train"]
        y_train = st.session_state["y_train"]
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]
    else:
        st.error("⚠️ Chưa có dữ liệu! Hãy chia dữ liệu trước.")
        return

    # 🌟 Chuẩn hóa dữ liệu
    X_train = X_train.reshape(-1, 28 * 28) / 255.0
    X_test = X_test.reshape(-1, 28 * 28) / 255.0

    st.header("⚙️ Chọn mô hình & Huấn luyện")

    # 📌 **Chọn mô hình**
    model_choice = st.selectbox("Chọn mô hình:", ["K-Means", "DBSCAN"])

    if model_choice == "K-Means":
        st.markdown("""
        - **🔹 K-Means** là thuật toán phân cụm phổ biến, chia dữ liệu thành K cụm dựa trên khoảng cách.
        - **Tham số cần chọn:**  
            - **n_clusters**: Số lượng cụm (k).  
        """)
        
        n_clusters = st.slider("n_clusters", 2, 20, 10)
        model = KMeans(n_clusters=n_clusters, random_state=42)
    
    elif model_choice == "DBSCAN":
        st.markdown("""
        - **🛠️ DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** là thuật toán phân cụm dựa trên mật độ.
        - **Tham số cần chọn:**  
            - **eps**: Bán kính lân cận.  
            - **min_samples**: Số lượng điểm tối thiểu để tạo cụm.  
        """)
        eps = st.slider("eps", 0.1, 10.0, 0.5)
        min_samples = st.slider("min_samples", 2, 20, 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)

    if st.button("Huấn luyện mô hình"):
        model.fit(X_train)
        labels = model.labels_
        st.success("✅ Huấn luyện thành công!")

        # Lưu mô hình vào session_state dưới dạng danh sách nếu chưa có
        if "models" not in st.session_state:
            st.session_state["models"] = []

        model_name = model_choice.lower().replace(" ", "_")

        existing_model = next((item for item in st.session_state["models"] if item["name"] == model_name), None)
        
        if existing_model:
            count = 1
            new_model_name = f"{model_name}_{count}"
            while any(item["name"] == new_model_name for item in st.session_state["models"]):
                count += 1
                new_model_name = f"{model_name}_{count}"
            model_name = new_model_name
            st.warning(f"⚠️ Mô hình được lưu với tên là: {model_name}")

        st.session_state["models"].append({"name": model_name, "model": model})
        st.write(f"🔹 Mô hình đã được lưu với tên: {model_name}")
        st.write(f"Tổng số mô hình hiện tại: {len(st.session_state['models'])}")

        st.write("📋 Danh sách các mô hình đã lưu:")
        model_names = [model["name"] for model in st.session_state["models"]]
        st.write(", ".join(model_names))
        


def ClusteringAlgorithms():
  

    st.title("🖊️ MNIST Classification App")

    ### **Phần 1: Hiển thị dữ liệu MNIST**
    
    ### **Phần 2: Trình bày lý thuyết về Decision Tree & SVM*
    
    # 1️⃣ Phần giới thiệu
    
    # === Sidebar để chọn trang ===
    # === Tạo Tabs ===
    tab1, tab2, tab3, tab4,tab5 = st.tabs(["📘 Lý thuyết K-means", "📘 Lý thuyết DBSCAN", "📘 Data" ,"⚙️ Huấn luyện", "🔢 Dự đoán"])

    with tab1:
        ly_thuyet_K_means()

    with tab2:
        ly_thuyet_DBSCAN()
    
    with tab3:
        data()
        
    with tab4:
       # plot_tree_metrics()
        
        
        
        split_data()
        train()
        
    
    with tab5:
        pass  





            
if __name__ == "__main__":
    ClusteringAlgorithms()