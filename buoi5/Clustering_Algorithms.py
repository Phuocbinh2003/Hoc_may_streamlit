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

# Tải dữ liệu MNIST từ OpenM




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
    ## 📌 **K-Means Clustering**
    **K-Means** là thuật toán **phân cụm không giám sát** phổ biến, giúp chia dữ liệu thành **K cụm** sao cho các điểm dữ liệu trong cùng một cụm có đặc trưng giống nhau nhất.

    ---

    ### 🔹 **Ý tưởng chính**
    1️⃣ **Chọn ngẫu nhiên \( K \) tâm cụm (centroids)** từ tập dữ liệu.  
    2️⃣ **Gán mỗi điểm dữ liệu vào cụm có tâm gần nhất**, sử dụng khoảng cách Euclidean:  
    """)

    # Công thức khoảng cách Euclidean
    st.latex(r"""
    d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}
    """)

    st.markdown("""
    3️⃣ **Cập nhật lại tâm cụm** bằng cách tính trung bình của các điểm trong cụm:
    """)

    # Công thức cập nhật tâm cụm
    st.latex(r"""
    \mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i
    """)

    st.markdown("""
    4️⃣ **Lặp lại quá trình trên** cho đến khi không có sự thay đổi hoặc đạt đến số lần lặp tối đa.

    ---

    ### 🔢 **Công thức tổng quát của K-Means**
    K-Means tối ưu hóa tổng bình phương khoảng cách từ mỗi điểm đến tâm cụm của nó:
    """)

    # Hàm mục tiêu của K-Means
    st.latex(r"""
    J = \sum_{k=1}^{K} \sum_{x_i \in C_k} ||x_i - \mu_k||^2
    """)

    st.markdown("""
    - **\( J \)**: Hàm mất mát (tổng bình phương khoảng cách).
    - **\( x_i \)**: Điểm dữ liệu thứ \( i \).
    - **\( \mu_k \)**: Tâm cụm thứ \( k \).
    - **\( C_k \)**: Nhóm các điểm thuộc cụm thứ \( k \).

    ---

    ### ✅ **Ưu điểm & ❌ Nhược điểm**
    ✅ **Ưu điểm:**
    - Đơn giản, dễ hiểu, tốc độ nhanh.
    - Hiệu quả trên tập dữ liệu lớn.
    - Dễ triển khai và mở rộng.

    ❌ **Nhược điểm:**
    - Cần xác định số cụm **\( K \)** trước.
    - Nhạy cảm với giá trị **outlier**.
    - Kết quả phụ thuộc vào vị trí khởi tạo ban đầu.

    ---

    ### 🔍 **Một số cải tiến của K-Means**
    - **K-Means++**: Cải thiện cách chọn tâm cụm ban đầu.
    - **Mini-batch K-Means**: Dùng mẫu nhỏ thay vì toàn bộ dữ liệu để tăng tốc độ.
    - **K-Medoids**: Thay vì trung bình, sử dụng điểm thực tế làm tâm cụm.

    📌 **K-Means được ứng dụng rộng rãi trong:** Phân tích khách hàng, nhận diện mẫu, nén ảnh, phân cụm văn bản, v.v.
    """)


    # 🔹 Định nghĩa hàm tính toán
    def euclidean_distance(a, b):
        return np.linalg.norm(a - b, axis=1)

    def generate_data(n_samples, n_clusters, cluster_std):
        np.random.seed(42)
        X = []
        centers = np.random.uniform(-10, 10, size=(n_clusters, 2))
        for c in centers:
            X.append(c + np.random.randn(n_samples // n_clusters, 2) * cluster_std)
        return np.vstack(X)

    def initialize_centroids(X, k):
        return X[np.random.choice(X.shape[0], k, replace=False)]

    def assign_clusters(X, centroids):
        return np.array([np.argmin(euclidean_distance(x, centroids)) for x in X])

    def update_centroids(X, labels, k):
        return np.array([X[labels == i].mean(axis=0) if len(X[labels == i]) > 0 else np.random.uniform(-10, 10, 2) for i in range(k)])

    # Giao diện Streamlit
    st.title("🎯 Minh họa thuật toán K-Means từng bước")

    num_samples_kmeans = st.slider("Số điểm dữ liệu", 50, 500, 200, step=10)
    cluster_kmeans = st.slider("Số cụm (K)", 2, 10, 3)
    spread_kmeans = st.slider("Độ rời rạc", 0.1, 2.0, 1.0)

    # if "X" not in st.session_state:
    #     st.session_state.X = generate_data(num_samples_kmeans, cluster_kmeans, spread_kmeans)

    # X = st.session_state.X

    # Kiểm tra và cập nhật dữ liệu khi tham số thay đổi
    if "data_params" not in st.session_state or st.session_state.data_params != (num_samples_kmeans, cluster_kmeans, spread_kmeans):
        st.session_state.data_params = (num_samples_kmeans, cluster_kmeans, spread_kmeans)
        st.session_state.X = generate_data(num_samples_kmeans, cluster_kmeans, spread_kmeans)
        st.session_state.centroids = initialize_centroids(st.session_state.X, cluster_kmeans)
        st.session_state.iteration = 0
        st.session_state.labels = assign_clusters(st.session_state.X, st.session_state.centroids)

    X = st.session_state.X


    if st.button("🔄 Reset"):
        st.session_state.X = generate_data(num_samples_kmeans, cluster_kmeans, spread_kmeans)
        st.session_state.centroids = initialize_centroids(st.session_state.X, cluster_kmeans)
        st.session_state.iteration = 0
        st.session_state.labels = assign_clusters(st.session_state.X, st.session_state.centroids)

    if st.button("🔄 Cập nhật vị trí tâm cụm"):
        st.session_state.labels = assign_clusters(X, st.session_state.centroids)
        new_centroids = update_centroids(X, st.session_state.labels, cluster_kmeans)
        
        # Kiểm tra hội tụ với sai số nhỏ
        if np.allclose(new_centroids, st.session_state.centroids, atol=1e-3):
            st.warning("⚠️ Tâm cụm không thay đổi đáng kể, thuật toán đã hội tụ!")
        else:
            st.session_state.centroids = new_centroids
            st.session_state.iteration += 1

    # 🔥 Thêm thanh trạng thái hiển thị tiến trình
    
    
    
    st.status(f"Lần cập nhật: {st.session_state.iteration} - Đang phân cụm...", state="running")
    st.markdown("### 📌 Tọa độ tâm cụm hiện tại:")
    num_centroids = st.session_state.centroids.shape[0]  # Số lượng tâm cụm thực tế
    centroid_df = pd.DataFrame(st.session_state.centroids, columns=["X", "Y"])
    centroid_df.index = [f"Tâm cụm {i}" for i in range(num_centroids)]  # Đảm bảo index khớp

    st.dataframe(centroid_df)
    
    
    
    
    # Vẽ biểu đồ
    fig, ax = plt.subplots(figsize=(6, 6))
    labels = st.session_state.labels
    centroids = st.session_state.centroids

    for i in range(cluster_kmeans):
        ax.scatter(X[labels == i][:, 0], X[labels == i][:, 1], label=f"Cụm {i}", alpha=0.6, edgecolors="k")

    ax.scatter(centroids[:, 0], centroids[:, 1], s=200, c="red", marker="X", label="Tâm cụm")
    ax.set_title(f"K-Means Clustering")
    ax.legend()

    st.pyplot(fig)


from sklearn.datasets import make_moons, make_blobs
from sklearn.cluster import DBSCAN

def ly_thuyet_DBSCAN():

  
    st.markdown(r"""
        ## 📌 **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**
        **DBSCAN** là một thuật toán phân cụm **không giám sát**, dựa trên **mật độ điểm dữ liệu**, giúp xác định các cụm có hình dạng bất kỳ và phát hiện nhiễu (outliers).  

        ---

        ### 🔹 **Ý tưởng chính của DBSCAN**
        1️⃣ **Xác định các điểm lõi (Core Points):** Nếu một điểm có ít nhất **min_samples** điểm lân cận trong bán kính **\(  varepsilon \)**, nó là một **điểm lõi**.  
        2️⃣ **Xác định các điểm biên (Border Points):** Là các điểm thuộc vùng lân cận của điểm lõi nhưng không đủ **min_samples**.  
        3️⃣ **Xác định nhiễu (Noise Points):** Các điểm không thuộc bất kỳ cụm nào.  
        4️⃣ **Mở rộng cụm:** Bắt đầu từ một điểm lõi, mở rộng cụm bằng cách thêm các điểm biên lân cận cho đến khi không còn điểm nào thoả mãn điều kiện.  

        ---

        ### 🔢 **Tham số quan trọng của DBSCAN**
        - **\( varepsilon \)** (eps): Bán kính tìm kiếm điểm lân cận.  
        - **min_samples**: Số lượng điểm tối thiểu trong **eps** để xác định một **core point**.  

        ---

        ### 📌 **Công thức khoảng cách trong DBSCAN**
        DBSCAN sử dụng **khoảng cách Euclidean** để xác định **điểm lân cận**, được tính bằng công thức:  
        """)

    st.latex(r"d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}")

    st.markdown(r"""
        Trong đó:  
        - \( d(p, q) \) là khoảng cách giữa hai điểm dữ liệu \( p \) và \( q \).  
        - \( p_i \) và \( q_i \) là tọa độ của điểm \( p \) và \( q \) trong không gian **n** chiều.  

        ---

        ### 🔢 **Cách hoạt động của DBSCAN**
        **Gọi \( N_{\varepsilon}(p) \) là tập hợp các điểm nằm trong bán kính \( \varepsilon \) của \( p \):**  
        """)
    st.markdown(r"""
        ### 🔢 **Cách hoạt động của DBSCAN**
        **Gọi** \( N_{\varepsilon}(p) \) **là tập hợp các điểm nằm trong bán kính** \( \varepsilon \) **của** \( p \):  
        """)

    st.latex(r"N_{\varepsilon}(p) = \{ q \in D \mid d(p, q) \leq \varepsilon \}")
    

    st.markdown(r"""
        - Nếu \( |N_{\varepsilon}(p)| \geq \) min_samples, thì **\( p \)** là một **core point**.  
        - Nếu **\( p \)** là **core point**, tất cả các điểm trong \( N_{\varepsilon}(p) \) sẽ được gán vào cùng một cụm.  
        - Nếu một điểm không thuộc cụm nào, nó được đánh dấu là **nhiễu**.  

        ---

        ### ✅ **Ưu điểm & ❌ Nhược điểm**
        ✅ **Ưu điểm:**  
        - Tự động tìm số cụm mà **không cần xác định trước \( K \)** như K-Means.  
        - Xử lý tốt **các cụm có hình dạng phức tạp**.  
        - Phát hiện **outlier** một cách tự nhiên.  

        ❌ **Nhược điểm:**  
        - Nhạy cảm với **tham số \( varepsilon \) và min_samples**.  
        - Không hoạt động tốt trên **dữ liệu có mật độ thay đổi**.  

        ---

        ### 📌 **Ứng dụng của DBSCAN**
        - **Phát hiện gian lận tài chính**.  
        - **Phân tích dữ liệu không gian (GIS, bản đồ)**.  
        - **Phát hiện bất thường (Anomaly Detection)**.  
    """)


    # Tiếp tục phần giao diện chạy DBSCAN


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
    # Tùy chỉnh tham số với key để tránh lỗi trùng ID
    
    dataset_type = st.radio("Chọn kiểu dữ liệu", ["Cụm Gauss", "Hai vòng trăng (Moons)"], key="dataset_type")
    

    num_samples_dbscan = st.slider("Số điểm dữ liệu", 50, 500, 200, step=10, key="num_samples_dbscan")
    noise_dbscan = st.slider("Mức nhiễu", 0.05, 1.0, 0.2, key="noise_dbscan")
    eps_dbscan = st.slider("Bán kính cụm (eps)", 0.1, 2.0, 0.5, step=0.1, key="eps_dbscan")
    min_samples_dbscan = st.slider("Số điểm tối thiểu để tạo cụm", 2, 20, 5, key="min_samples_dbscan")

    # Nút Reset để tạo lại dữ liệu
    if st.button("🔄 Reset", key="reset_dbscan"):
        st.session_state.X = generate_data(num_samples_dbscan, noise_dbscan, dataset_type)
        st.session_state.labels = np.full(num_samples_dbscan, -1)  # Chưa có cụm nào

    # Kiểm tra dữ liệu trong session_state
    if "X" not in st.session_state:
        st.session_state.X = generate_data(num_samples_dbscan, noise_dbscan, dataset_type)
        st.session_state.labels = np.full(num_samples_dbscan, -1)

    X = st.session_state.X

    # Nút chạy DBSCAN
    if st.button("➡️ Chạy DBSCAN"):
        st.session_state.labels = run_dbscan(X, eps_dbscan, min_samples_dbscan)

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

    ax.set_title(f"Kết quả DBSCAN (eps={eps_dbscan}, min_samples={min_samples_dbscan})")
    ax.legend()

    # Hiển thị biểu đồ
    st.pyplot(fig)




# Hàm vẽ biểu đồ
import streamlit as st
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.stats import mode  

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

    if "X_train" in st.session_state:
        st.write("📌 Dữ liệu train/test đã sẵn sàng để sử dụng!")

def train():
    st.header("⚙️ Chọn mô hình & Huấn luyện")

    # Kiểm tra dữ liệu trước khi train
    if "X_train" not in st.session_state:
        st.warning("⚠️ Vui lòng chia dữ liệu trước khi train!")
        return

    X_train = st.session_state["X_train"]
    y_train = st.session_state["y_train"]

    # 🌟 **Chuẩn hóa dữ liệu**
    X_train = X_train.reshape(-1, 28 * 28) / 255.0

    # 📌 **Chọn mô hình**
    model_choice = st.selectbox("Chọn mô hình:", ["K-Means", "DBSCAN"])

    if model_choice == "K-Means":
        st.markdown("🔹 **K-Means**: Thuật toán phân cụm chia dữ liệu thành K nhóm dựa trên khoảng cách.")

        n_clusters = st.slider("🔢 Chọn số cụm (K):", 2, 20, 10)

        # 📉 Giảm chiều dữ liệu bằng PCA trước khi huấn luyện
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train)

        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    elif model_choice == "DBSCAN":
        st.markdown("🛠️ **DBSCAN**: Thuật toán phân cụm dựa trên mật độ.")

        eps = st.slider("📏 Bán kính lân cận (eps):", 0.1, 10.0, 0.5)
        min_samples = st.slider("👥 Số điểm tối thiểu trong cụm:", 2, 20, 5)

        # 📉 Giảm chiều dữ liệu bằng PCA trước khi huấn luyện
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train)

        model = DBSCAN(eps=eps, min_samples=min_samples)

    if st.button("🚀 Huấn luyện mô hình"):
        model.fit(X_train_pca)
        st.success("✅ Huấn luyện thành công!")

        if model_choice == "K-Means":
            labels = model.labels_

            # 🔄 Ánh xạ nhãn cụm với nhãn thực tế
            label_mapping = {}
            for i in range(n_clusters):
                mask = labels == i
                if np.sum(mask) > 0:
                    most_common_label = mode(y_train[mask], keepdims=True).mode[0]  
                    label_mapping[i] = most_common_label

            # 🎯 Chuyển nhãn cụm thành nhãn thực
            predicted_labels = np.array([label_mapping[label] for label in labels])

            # ✅ Tính độ chính xác
            accuracy = np.mean(predicted_labels == y_train)
            st.write(f"🎯 **Độ chính xác của mô hình:** `{accuracy * 100:.2f}%`")

        # 🔍 Lưu mô hình vào session_state
        if "models" not in st.session_state:
            st.session_state["models"] = []

        model_name = model_choice.lower().replace(" ", "_")

        # Kiểm tra tên để tránh trùng lặp
        count = 1
        new_model_name = model_name
        while any(m["name"] == new_model_name for m in st.session_state["models"]):
            new_model_name = f"{model_name}_{count}"
            count += 1

        st.session_state["models"].append({"name": new_model_name, "model": model})

        st.write(f"🔹 **Mô hình đã được lưu với tên:** `{new_model_name}`")
        st.write(f"📋 **Danh sách các mô hình:** {[m['name'] for m in st.session_state['models']]}")




import streamlit as st
import numpy as np
import random
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
from sklearn.decomposition import PCA

def preprocess_canvas_image(canvas_result):
    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data[:, :, 0].astype(np.uint8))
        img = img.resize((28, 28)).convert("L")  # Resize và chuyển thành grayscale
        img = np.array(img, dtype=np.float32) / 255.0  # Chuẩn hóa về [0, 1]
        return img.reshape(1, -1)  # Chuyển thành vector 1D
    return None


def du_doan():
    st.header("✍️ Vẽ dữ liệu để dự đoán cụm")

    # Kiểm tra danh sách mô hình đã huấn luyện
    if "models" not in st.session_state or not st.session_state["models"]:
        st.warning("⚠️ Không có mô hình nào được lưu! Hãy huấn luyện trước.")
        return

    # Lấy danh sách mô hình đã lưu
    model_names = [model["name"] for model in st.session_state["models"]]

    # 📌 Chọn mô hình
    model_option = st.selectbox("🔍 Chọn mô hình đã huấn luyện:", model_names)
    model = next(m["model"] for m in st.session_state["models"] if m["name"] == model_option)

    # 🆕 Cập nhật key cho canvas khi nhấn "Tải lại"
    if "key_value" not in st.session_state:
        st.session_state.key_value = str(random.randint(0, 1000000))

    if st.button("🔄 Tải lại"):
        st.session_state.key_value = str(random.randint(0, 1000000))
        st.rerun()

    # ✍️ Vẽ dữ liệu
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=10,
        stroke_color="white",
        background_color="black",
        height=150,
        width=150,
        drawing_mode="freedraw",
        key=st.session_state.key_value,
        update_streamlit=True
    )

    if st.button("Dự đoán cụm"):
        img = preprocess_canvas_image(canvas_result)

        if img is not None:
            X_train = st.session_state["X_train"]
            # Hiển thị ảnh sau xử lý
            st.image(Image.fromarray((img.reshape(28, 28) * 255).astype(np.uint8)), caption="Ảnh sau xử lý", width=100)
            
            pca = PCA(n_components=2)
            pca.fit(X_train)
            img_reduced = pca.transform(img.squeeze().reshape(1, -1))  # Sửa lỗi

            # Dự đoán với K-Means hoặc DBSCAN
            if isinstance(model, KMeans):
                predicted_cluster = model.predict(img_reduced)[0]  # Dự đoán từ ảnh đã PCA
                st.subheader(f"🔢 Cụm dự đoán: {predicted_cluster}")

            elif isinstance(model, DBSCAN):
                model.fit(X_train)  # Fit trước với tập huấn luyện
                predicted_cluster = model.fit_predict(img_reduced)[0]
                if predicted_cluster == -1:
                    st.subheader("⚠️ Điểm này không thuộc cụm nào!")
                else:
                    st.subheader(f"🔢 Cụm dự đoán: {predicted_cluster}")

        else:
            st.error("⚠️ Hãy vẽ một số trước khi bấm Dự đoán!")





def ClusteringAlgorithms():
  

    st.title("🖊️ MNIST Classification App")

    ### **Phần 1: Hiển thị dữ liệu MNIST**
    
    ### **Phần 2: Trình bày lý thuyết về Decision Tree & SVM*
    
    # 1️⃣ Phần giới thiệ
    
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
       
        
        
        
        split_data()
        train()
        
    
    with tab5:
        
        du_doan() 





            
if __name__ == "__main__":
    ClusteringAlgorithms()
