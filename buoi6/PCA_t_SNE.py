import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml
def explain_pca():
    st.markdown("## 🧠 PCA ")

    st.markdown("""
    **PCA (Phân tích thành phần chính)** là một phương pháp giúp giảm số chiều của dữ liệu mà vẫn giữ được thông tin quan trọng nhất.  
    Hãy tưởng tượng bạn có một tập dữ liệu nhiều chiều (nhiều cột), nhưng bạn muốn biểu diễn nó trong không gian 2D hoặc 3D để dễ hiểu hơn. PCA giúp bạn làm điều đó!  

    ### 🔹 **Ví dụ trực quan**:
    Hãy tưởng tượng bạn có một tập dữ liệu gồm nhiều điểm phân bố theo một đường chéo trong không gian 2D:
    """)

   
    st.image("buoi6/img9.png",use_container_width ="auto", caption="dữ liệu mô phỏng")   


    st.markdown(r"""
    ## 📌 PCA - Giải thích Trực Quan  
    Dữ liệu này có sự phân tán rõ ràng theo một hướng chính. PCA sẽ tìm ra hướng đó để biểu diễn dữ liệu một cách tối ưu.

    ---

    ### 🔹 **Các bước thực hiện PCA**

    1️⃣ **Tìm điểm trung tâm (mean vector)**  
    - Trước tiên, tính giá trị trung bình của từng đặc trưng (feature) trong tập dữ liệu.  
    - Vector trung bình này giúp xác định "trung tâm" của dữ liệu.  
    $$ 
    \mu = \frac{1}{n} \sum_{i=1}^{n} x_i 
    $$  
    - Trong đó:
        - \( n \) là số lượng mẫu dữ liệu.
        - $$ ( x_i )$$  là từng điểm dữ liệu.
        
        
    """)  
    st.image("buoi6/img3.png",use_container_width ="auto", caption="Trung bình dữ liệu(machinelearningcoban.com)")   

        
    st.markdown(r"""
    2️⃣ **Dịch chuyển dữ liệu về gốc tọa độ**  
    - Để đảm bảo phân tích chính xác hơn, ta dịch chuyển dữ liệu sao cho trung tâm của nó nằm tại gốc tọa độ bằng cách trừ đi vector trung bình:  
    $$ 
    X_{\text{norm}} = X - \mu
    $$  
    - Khi đó, dữ liệu sẽ có giá trị trung bình bằng 0.
    """)
    st.image("buoi6/img4.png", caption="Dịch chuyển dữ liệu về gốc tọa độ(machinelearningcoban.com)") 
    st.markdown(r"""
    3️⃣ **Tính ma trận hiệp phương sai**  
    - Ma trận hiệp phương sai giúp đo lường mức độ biến thiên giữa các đặc trưng:  
    $$ 
    C = \frac{1}{n} X_{\text{norm}}^T X_{\text{norm}}
    $$  
    - Ý nghĩa:
        - Nếu phần tử $$ ( C_{ij} )$$  có giá trị lớn → Hai đặc trưng \( i \) và \( j \) có mối tương quan mạnh.
        - Nếu $$ ( C_{ij} )$$  gần 0 → Hai đặc trưng không liên quan nhiều.
        
        
    

    4️⃣ **Tìm các hướng quan trọng nhất**  
    - Tính trị riêng (eigenvalues) và vector riêng (eigenvectors) từ ma trận hiệp phương sai:  
    $$ 
    C v = \lambda v
    $$  
    - Trong đó:
        - \( v \) là vector riêng (eigenvector) - đại diện cho các hướng chính của dữ liệu.
        - $$\lambda $$ là trị riêng (eigenvalue) - thể hiện độ quan trọng của từng hướng.
    - Vector riêng có trị riêng lớn hơn sẽ mang nhiều thông tin quan trọng hơn.
    
    """)
    
    st.markdown(r"""
    5️⃣ **Chọn số chiều mới và tạo không gian con**  
    - Chọn \( K \) vector riêng tương ứng với \( K \) trị riêng lớn nhất để tạo ma trận $$( U_K )$$:  
    $$ 
    U_K = [v_1, v_2, ..., v_K]
    $$  
    - Các vector này tạo thành hệ trực giao và giúp ta biểu diễn dữ liệu tối ưu trong không gian mới.
    """)
    st.image("buoi6/img5.png",caption="Chọn không gian con với các vector riêng(machinelearningcoban.com)") 
    st.markdown(r"""
    6️⃣ **Chiếu dữ liệu vào không gian mới**  
    - Biểu diễn dữ liệu trong hệ trục mới bằng cách nhân dữ liệu chuẩn hóa với ma trận $$( U_K )$$:  
    $$ 
    X_{\text{new}} = X_{\text{norm}} U_K
    $$  
    - Dữ liệu mới $$ X_{\text{new}} $$ có số chiều ít hơn nhưng vẫn giữ được nhiều thông tin quan trọng.
    """)
    st.image("buoi6/img6.png", caption="Chiếu dữ liệu vào không gian mới(machinelearningcoban.com)") 

    st.markdown(r"""
    7️⃣ **Dữ liệu mới chính là tọa độ của các điểm trong không gian mới.**  
    - Mỗi điểm dữ liệu giờ đây được biểu diễn bằng các thành phần chính thay vì các đặc trưng ban đầu.

    """)
    st.image("buoi6/img7.png", caption="Dữ liệu mới sau PCA(machinelearningcoban.com)") 
    


    st.markdown("""
    **🔹 Kết quả:**  
    
    
    
    
    
    - Trục đỏ là hướng mà PCA tìm ra.  
    - Nếu chọn 1 trục chính, ta có thể chiếu dữ liệu lên nó để giảm chiều.  
      
    Nhờ đó, chúng ta có thể biểu diễn dữ liệu một cách gọn gàng hơn mà không mất quá nhiều thông tin!  
    """)
    # st.image("buoi6/img1.png")  # Đường dẫn cần đúng
    st.markdown("""
        ### ✅ **Ưu điểm của PCA**  
        - **Giảm chiều dữ liệu hiệu quả**: PCA giúp giảm số chiều của dữ liệu mà vẫn giữ lại phần lớn thông tin quan trọng.  
        - **Tăng tốc độ xử lý**: Khi số chiều giảm, các mô hình học máy sẽ chạy nhanh hơn và yêu cầu ít tài nguyên hơn.  
        - **Giảm nhiễu**: PCA có thể loại bỏ các thành phần nhiễu bằng cách giữ lại các thành phần chính có phương sai cao.  
        - **Trực quan hóa dữ liệu**: PCA giúp hiển thị dữ liệu nhiều chiều dưới dạng 2D hoặc 3D để con người dễ quan sát hơn.  

        ---

        ### ❌ **Nhược điểm của PCA**  
        - **Mất thông tin**: PCA chọn những thành phần có phương sai cao nhất, có thể làm mất thông tin quan trọng.  
        - **Không phải lúc nào cũng hiệu quả**: PCA chỉ hoạt động tốt khi dữ liệu có cấu trúc tuyến tính. Với dữ liệu phi tuyến tính, t-SNE có thể tốt hơn.  
        - **Khó diễn giải**: Sau khi giảm chiều, các thành phần chính không còn giữ nguyên ý nghĩa gốc, khiến việc hiểu dữ liệu trở nên khó khăn hơn.  
        - **Ảnh hưởng bởi dữ liệu đầu vào**: PCA nhạy cảm với thang đo dữ liệu. Nếu dữ liệu chưa được chuẩn hóa, kết quả có thể bị méo mó.  
        """)

    
    
    
    
    

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
def explain_tsne():
    st.markdown(r"""
    ## 🌌 t-Distributed Stochastic Neighbor Embedding (t-SNE)
    t-SNE là một phương pháp giảm chiều mạnh mẽ, giúp hiển thị dữ liệu đa chiều trên mặt phẳng 2D hoặc không gian 3D bằng cách bảo toàn mối quan hệ giữa các điểm gần nhau.

    ---
    """)
    st.markdown(r"""
    # 🔹 Nguyên lý hoạt động của t-SNE

    ## 1️⃣ Tính xác suất lân cận trong không gian gốc  
    
    Trong không gian ban đầu (high-dimensional space), t-SNE sử dụng phân phối Gaussian để xác định mức độ gần gũi giữa các điểm dữ liệu.  
    
        """)
    st.image("buoi6/download (10).png", caption="Minh họa quá trình t-SNE", use_column_width=True)
    st.markdown(r"""
    - Với mỗi điểm \( x_i \), xác suất có điều kiện giữa \( x_i \) và \( x_j \) được tính theo công thức:  
    \[
    p_{j|i} = \frac{\exp(-\| x_i - x_j \|^2 / 2\sigma^2)}{\sum_{k \neq i} \exp(-\| x_i - x_k \|^2 / 2\sigma^2)}
    \]
    
    Trong đó:  
    - \( \sigma \) là độ lệch chuẩn (bandwidth) của Gaussian Kernel, điều chỉnh phạm vi ảnh hưởng của mỗi điểm.  
    - Xác suất này phản ánh mức độ gần gũi của các điểm dữ liệu trong không gian ban đầu.  

    **Ý nghĩa:**  
    - Nếu hai điểm gần nhau, xác suất \( p_{j|i} \) sẽ cao.  
    - Ngược lại, nếu hai điểm xa nhau, xác suất này sẽ nhỏ.  

    ## 2️⃣ Tính xác suất trong không gian giảm chiều (2D/3D)  
    Sau khi chọn số chiều mục tiêu (thường là 2D hoặc 3D để trực quan hóa), t-SNE xây dựng một phân phối xác suất mới dựa trên tọa độ của điểm dữ liệu trong không gian giảm chiều. Tuy nhiên, thay vì sử dụng phân phối Gaussian, t-SNE sử dụng phân phối **t-Student với một bậc tự do**:  

    \[
    q_{j|i} = \frac{(1 + \| y_i - y_j \|^2)^{-1}}{\sum_{k \neq i} (1 + \| y_i - y_k \|^2)^{-1}}
    \]

    **Tại sao dùng t-Student thay vì Gaussian?**  
    - Phân phối t-Student có đuôi dài hơn so với Gaussian. Điều này giúp duy trì khoảng cách xa giữa các điểm, tránh hiện tượng co cụm dữ liệu quá mức.  
    - Nhờ đó, các cụm dữ liệu tách biệt rõ ràng hơn, giúp trực quan hóa trở nên trực quan hơn.  

    ## 3️⃣ Giảm thiểu sai số giữa hai phân phối \( P \) và \( Q \)  
    Mục tiêu của t-SNE là làm cho phân phối trong không gian mới **giống với phân phối trong không gian ban đầu**. Để đo độ chênh lệch giữa hai phân phối này, t-SNE sử dụng hàm mất mát **Kullback-Leibler divergence (KL divergence)**:  
    """)
    
    st.image("buoi6/download (11).png", caption="Minh họa quá trình t-SNE", use_column_width=True)
    st.markdown(r"""
    \[
    KL(P||Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}
    \]

    Trong đó:  
    - \( P \) là phân phối trong không gian gốc.  
    - \( Q \) là phân phối trong không gian t-SNE.  

    **Ý nghĩa:**  
    - Nếu \( P \) và \( Q \) giống nhau, KL divergence sẽ nhỏ.  
    - t-SNE liên tục cập nhật vị trí của các điểm \( y_i \) để giảm KL divergence, giúp bảo toàn cấu trúc cục bộ của dữ liệu.  

    ---

    ## 📊 Trực quan hóa quá trình t-SNE  
    Hình dưới đây minh họa cách dữ liệu được biến đổi từ không gian nhiều chiều về không gian thấp hơn trong quá trình tối ưu hóa:  
    """)

    st.image("buoi6/download (12).png", caption="Minh họa quá trình t-SNE", use_column_width=True)

    st.markdown(r"""
    ---

    🔥 **Tóm tắt:**  
    - t-SNE chuyển đổi dữ liệu từ không gian nhiều chiều sang không gian thấp hơn bằng cách tối ưu hóa sự tương đồng giữa các điểm.  
    - Nó sử dụng **Gaussian Kernel** trong không gian gốc và **t-Student distribution** trong không gian giảm chiều.  
    - Quá trình tối ưu hóa dựa trên **KL divergence** giúp duy trì cấu trúc cục bộ của dữ liệu, làm cho các cụm điểm trở nên rõ ràng hơn.  
    """)



    # Kết luận
    st.markdown(r"""
    ---
    
    ### ✅ **Ưu điểm của t-SNE**
    - Tạo cụm dữ liệu rõ ràng, dễ quan sát.
    - Giữ được mối quan hệ phi tuyến tính trong dữ liệu.
    - Hiệu quả trong việc **trực quan hóa dữ liệu nhiều chiều**.

    ### ❌ **Nhược điểm của t-SNE**
    - Chạy chậm hơn PCA, đặc biệt với dữ liệu lớn.
    - Nhạy cảm với tham số **perplexity** (nếu chọn sai có thể gây méo mó dữ liệu).
    - Không bảo toàn khoảng cách toàn cục (chỉ bảo toàn quan hệ giữa các điểm gần nhau).

    ---

    📌 **Ghi nhớ:**  
    - t-SNE **không phù hợp cho các bài toán giảm chiều trước khi huấn luyện mô hình học máy**, vì nó **không bảo toàn cấu trúc tổng thể của dữ liệu**.  
    - Nó chủ yếu dùng để **trực quan hóa dữ liệu nhiều chiều**, giúp con người hiểu rõ hơn về cấu trúc của dữ liệu.

    🎯 **Ứng dụng thực tế:**  
    - Trực quan hóa dữ liệu hình ảnh (MNIST, CIFAR).  
    - Phân cụm dữ liệu văn bản.  
    - Phát hiện bất thường (anomaly detection).  
    """)

import mlflow
import os
import time
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def input_mlflow():
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/Phuocbinh2003/Hoc_may_python.mlflow"
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI
    os.environ["MLFLOW_TRACKING_USERNAME"] = "Phuocbinh2003"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "c1495823c8f9156923b06f15899e989db7e62052"
    mlflow.set_experiment("PCA_t-SNE")

import streamlit as st
import numpy as np
import plotly.express as px
import mlflow
import time
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def thi_nghiem():
    st.title("📉 Giảm chiều dữ liệu MNIST với PCA & t-SNE")

    # Load dữ liệu
    Xmt = np.load("buoi4/X.npy")
    ymt = np.load("buoi4/y.npy")
    X = Xmt.reshape(Xmt.shape[0], -1)
    y = ymt.reshape(-1)

    # Tùy chọn thuật toán
    method = st.radio("Chọn phương pháp giảm chiều", ["PCA", "t-SNE"])
    n_components = st.slider("Chọn số chiều giảm xuống", 2, min(X.shape[1], 50), 3)
    
    # Chọn trực quan hóa 2D hoặc 3D
    visualization_dim = st.radio("Chọn cách trực quan hóa", ["2D", "3D"])
    vis_components = 3 if visualization_dim == "3D" else 2
    
    # Thanh trượt chọn số lượng mẫu
    num_samples = st.slider("Chọn số lượng mẫu MNIST sử dụng:", min_value=1000, max_value=60000, value=5000, step=1000)
    
    X_subset, y_subset = X[:num_samples], y[:num_samples]
    
    run_name = st.text_input("🔹 Nhập tên Run:", "Default_Run")
    st.session_state["run_name"] = run_name if run_name else "default_run"
    
    if st.button("🚀 Chạy giảm chiều"):
        with st.spinner("Đang xử lý..."):
            mlflow.start_run(run_name=st.session_state["run_name"])
            mlflow.log_param("method", method)
            mlflow.log_param("n_components", n_components)
            mlflow.log_param("num_samples", num_samples)
            mlflow.log_param("original_dim", X.shape[1])
            mlflow.log_param("visualization_dim", vis_components)

            if method == "t-SNE":
                perplexity = min(30, num_samples - 1)
                mlflow.log_param("perplexity", perplexity)
                reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
            else:
                reducer = PCA(n_components=n_components)
            
            start_time = time.time()
            X_reduced = reducer.fit_transform(X_subset)
            elapsed_time = time.time() - start_time
            mlflow.log_metric("elapsed_time", elapsed_time)
            
            if method == "PCA":
                explained_variance = np.sum(reducer.explained_variance_ratio_)
                mlflow.log_metric("explained_variance_ratio", explained_variance)
            elif method == "t-SNE" and hasattr(reducer, "kl_divergence_"):
                mlflow.log_metric("KL_divergence", reducer.kl_divergence_)
            
            # Hiển thị kết quả
            if vis_components == 2:
                fig = px.scatter(x=X_reduced[:, 0], y=X_reduced[:, 1], color=y_subset.astype(str),
                                 title=f"{method} giảm chiều xuống {n_components}D (Trực quan hóa 2D)",
                                 labels={'x': "Thành phần 1", 'y': "Thành phần 2"})
            else:
                fig = px.scatter_3d(x=X_reduced[:, 0], y=X_reduced[:, 1], z=X_reduced[:, 2],
                                     color=y_subset.astype(str),
                                     title=f"{method} giảm chiều xuống {n_components} (Trực quan hóa 3D)",
                                     labels={'x': "Thành phần 1", 'y': "Thành phần 2", 'z': "Thành phần 3"})
            
            st.plotly_chart(fig)
            
            # Lưu kết quả vào MLflow
            os.makedirs("logs", exist_ok=True)
            fig_path = f"logs/{method}_{n_components}D.png"
            fig.write_image(fig_path)
            mlflow.log_artifact(fig_path)
            
            np.save(f"logs/{method}_X_reduced.npy", X_reduced)
            mlflow.log_artifact(f"logs/{method}_X_reduced.npy")
            
            mlflow.end_run()
            st.success(f"✅ Đã log dữ liệu cho Experiments PCA_t-SNE với Name: **Train_{st.session_state['run_name']}**!")
            st.markdown(f"### 🔗 [Truy cập MLflow DAGsHub]({st.session_state['mlflow_url']})")
            st.success("Hoàn thành!")

from datetime import datetime    
import streamlit as st
import mlflow
from datetime import datetime

def show_experiment_selector():
    st.title("📊 MLflow")
    
    # Kết nối với DAGsHub MLflow Tracking
    mlflow.set_tracking_uri("https://dagshub.com/Phuocbinh2003/Hoc_may_python.mlflow")
    
    # Lấy danh sách tất cả experiments
    experiment_name = "PCA_t-SNE"
    experiments = mlflow.search_experiments()
    selected_experiment = next((exp for exp in experiments if exp.name == experiment_name), None)

    if not selected_experiment:
        st.error(f"❌ Experiment '{experiment_name}' không tồn tại!")
        return

    st.subheader(f"📌 Experiment: {experiment_name}")
    st.write(f"**Experiment ID:** {selected_experiment.experiment_id}")
    st.write(f"**Trạng thái:** {'Active' if selected_experiment.lifecycle_stage == 'active' else 'Deleted'}")
    st.write(f"**Vị trí lưu trữ:** {selected_experiment.artifact_location}")

    # Lấy danh sách runs trong experiment
    runs = mlflow.search_runs(experiment_ids=[selected_experiment.experiment_id])

    if runs.empty:
        st.warning("⚠ Không có runs nào trong experiment này.")
        return

    st.write("### 🏃‍♂️ Các Runs gần đây:")
    
    
    run_info = []
    for _, run in runs.iterrows():
        run_id = run["run_id"]
        run_tags = mlflow.get_run(run_id).data.tags
        run_name = run_tags.get("mlflow.runName", f"Run {run_id[:8]}")  # Lấy từ tags
        run_info.append((run_name, run_id))
    
    # Tạo dictionary để map run_name -> run_id
    run_name_to_id = dict(run_info)
    run_names = list(run_name_to_id.keys())
    
    # Chọn run theo run_name
    selected_run_name = st.selectbox("🔍 Chọn một run:", run_names)
    selected_run_id = run_name_to_id[selected_run_name]

    # Hiển thị thông tin chi tiết của run được chọn
    selected_run = mlflow.get_run(selected_run_id)

    if selected_run:
        st.subheader(f"📌 Thông tin Run: {selected_run_name}")
        st.write(f"**Run ID:** {selected_run_id}")
        st.write(f"**Trạng thái:** {selected_run.info.status}")
        
        start_time_ms = selected_run.info.start_time  # Thời gian lưu dưới dạng milliseconds
        if start_time_ms:
            start_time = datetime.fromtimestamp(start_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
        else:
            start_time = "Không có thông tin"
        
        st.write(f"**Thời gian chạy:** {start_time}")

        # Hiển thị thông số đã log
        params = selected_run.data.params
        metrics = selected_run.data.metrics

        if params:
            st.write("### ⚙️ Parameters:")
            st.json(params)

        if metrics:
            st.write("### 📊 Metrics:")
            st.json(metrics)

        # Kiểm tra và hiển thị dataset artifact
        dataset_path = f"{selected_experiment.artifact_location}/{selected_run_id}/artifacts/dataset.npy"
        st.write("### 📂 Dataset:")
        st.write(f"📥 [Tải dataset]({dataset_path})")
    else:
        st.warning("⚠ Không tìm thấy thông tin cho run này.")

        
        
        
          
import mlflow
import os
from mlflow.tracking import MlflowClient
def pca_tsne():
    #st.title("🚀 MLflow DAGsHub Tracking với Streamlit")
    
    
    
    tab1, tab2, tab3,tab4 = st.tabs(["📘 Lý thuyết PCA", "📘 Lý thuyết t-NSE", "📘 Giảm chiều","🔥 Mlflow"] )

    with tab1:
        explain_pca()

    with tab2:
        explain_tsne()
    
    with tab3:
        input_mlflow()
        thi_nghiem()
    with tab4:
        show_experiment_selector()


if __name__ == "__main__":
    pca_tsne()
