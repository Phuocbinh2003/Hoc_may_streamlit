import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import os
import time
from datetime import datetime
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import plotly.express as px

# ======================================
# PHẦN 1: LÝ THUYẾT NEURAL NETWORK
# ======================================
def explain_nn():
    st.markdown("""
    ## 🧠 Neural Network Cơ Bản

    **Neural Network (Mạng nơ-ron nhân tạo - ANN)** là một mô hình tính toán lấy cảm hứng từ cấu trúc và hoạt động của não bộ con người. Mạng bao gồm nhiều nơ-ron nhân tạo kết nối với nhau thành các lớp (layers), giúp mô hình học và nhận diện các mẫu trong dữ liệu.

    ### 🔰 Kiến trúc cơ bản:
    """)
    
    st.markdown("""
    ### 📌 Cấu trúc của một mạng nơ-ron nhân tạo gồm ba loại lớp chính:
    1. **Input Layer**: Lớp tiếp nhận dữ liệu đầu vào.
    2. **Hidden Layers**: Xử lý thông tin thông qua các trọng số (weights) và hàm kích hoạt.
    3. **Output Layer**: Lớp đưa ra kết quả dự đoán.
    """)
    
    st.image("buoi7/img1.webp", caption="", use_container_width=True)
    
    st.markdown("""
    Mỗi nơ-ron trong một lớp nhận tín hiệu từ các nơ-ron lớp trước, nhân với trọng số (weights), cộng với bias, rồi đưa vào một hàm kích hoạt để quyết định tín hiệu truyền đi.
    
    ### 📌 Công thức toán học trong Neural Network:
    Với một nơ-ron, giá trị đầu ra được tính như sau:
    
    \[ z = \sum_{i=1}^{n} w_i x_i + b \]
    
    Trong đó:
    - \( x_i \) là đầu vào (input features)
    - \( w_i \) là trọng số (weights)
    - \( b \) là bias
    - \( z \) là tổng có trọng số
    
    Sau đó, giá trị \( z \) đi qua hàm kích hoạt \( \sigma(z) \) để tạo đầu ra:
    
    \[ a = \sigma(z) \]
    
    Các hàm kích hoạt phổ biến sẽ được trình bày trong phần tiếp theo.
    """)
    
    st.markdown("""
    ### 🎯 Hàm Kích Hoạt (Activation Functions)
    Hàm kích hoạt giúp mạng học được các tính phi tuyến tính, nhờ đó có thể mô hình hóa các mối quan hệ phức tạp.
    """)
    
    st.image("buoi7/img2.png", caption="", use_container_width=True)
    
    st.markdown("""
    - **Sigmoid:** Chuyển đổi giá trị đầu vào thành khoảng từ 0 đến 1, phù hợp cho bài toán phân loại nhị phân.
      \[ \sigma(z) = \frac{1}{1 + e^{-z}} \]
    
    - **Tanh (Hyperbolic Tangent):** Đầu ra nằm trong khoảng từ -1 đến 1, giúp xử lý dữ liệu có cả giá trị dương và âm.
      \[ \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} \]
    
    - **ReLU (Rectified Linear Unit):** Nếu đầu vào âm thì bằng 0, còn nếu dương thì giữ nguyên giá trị.
      \[ ReLU(z) = \max(0, z) \]
    """)
    
    st.markdown("""
    ### 🔄 Quá trình huấn luyện Neural Network:
    Mạng nơ-ron học bằng cách cập nhật các trọng số thông qua hai giai đoạn chính:
    
    1. **Lan truyền thuận (Forward Propagation):**
       - Input đi qua từng lớp nơ-ron, tính toán đầu ra:
         \[ a^{(l)} = \sigma(W^{(l)} a^{(l-1)} + b^{(l)}) \]
    
    2. **Tính toán loss:**
       - Hàm mất mát đo lường sai số giữa dự đoán và thực tế.
       - Ví dụ: Mean Squared Error (MSE) cho bài toán hồi quy:
         \[ L = \frac{1}{N} \sum (y_{true} - y_{pred})^2 \]
       - Cross-Entropy Loss cho bài toán phân loại:
         \[ L = - \sum y_{true} \log(y_{pred}) \]
    
    3. **Lan truyền ngược (Backpropagation):**
       - Tính đạo hàm của hàm mất mát theo trọng số.
       - Sử dụng thuật toán tối ưu để cập nhật trọng số.
    
    4. **Tối ưu hóa:**
       - **Gradient Descent:** Cập nhật trọng số bằng cách đi theo hướng giảm của gradient.
       - **Momentum:** Thêm động lượng giúp tối ưu nhanh hơn.
       - **Adam (Adaptive Moment Estimation):** Kết hợp Momentum và RMSprop để đạt hiệu suất tối ưu.
    
    ### 🔍 Kết Luận
    Neural Network là một mô hình mạnh mẽ trong Machine Learning và Deep Learning, có thể học được các đặc trưng phức tạp từ dữ liệu. Hiểu rõ cách hoạt động giúp ta tối ưu hóa mô hình để đạt hiệu suất cao hơn.
    """)
    
    
    
    
    



# ======================================

def input_mlflow():
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/Phuocbinh2003/Hoc_may_python.mlflow"
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI
    os.environ["MLFLOW_TRACKING_USERNAME"] = "Phuocbinh2003"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "c1495823c8f9156923b06f15899e989db7e62052"
    mlflow.set_experiment("NN")

import streamlit as st
import numpy as np
import time
import mlflow
import mlflow.keras
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split, StratifiedKFold
from mlflow.models.signature import infer_signature

# Load dữ liệu MNIST
def load_mnist_data():
    X = np.load("buoi4/X.npy")
    y = np.load("buoi4/y.npy")
    return X, y

def split_data():
    st.title("📌 Chia dữ liệu Train/Test")
    
    # Đọc dữ liệu
    X, y = load_mnist_data()
    total_samples = X.shape[0]
    
    # Thanh kéo chọn số lượng ảnh để train
    num_samples = st.slider("📌 Chọn số lượng ảnh để train:", 1000, total_samples, 10000)
    
    # Thanh kéo chọn tỷ lệ Train/Test
    test_size = st.slider("📌 Chọn % dữ liệu Test", 10, 50, 20)
    train_size = 100 - test_size
    val_size = st.slider("📌 Chọn % dữ liệu Validation (trong Train)", 0, 50, 15)
    
    st.write(f"📌 **Tỷ lệ phân chia:** Test={test_size}%, Validation={val_size}%, Train={train_size - val_size}%")
    
    if st.button("✅ Xác nhận & Lưu"):
        X_selected, _, y_selected, _ = train_test_split(X, y, train_size=num_samples, stratify=y, random_state=42)
        X_train_full, X_test, y_train_full, y_test = train_test_split(X_selected, y_selected, test_size=test_size/100, stratify=y_selected, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=val_size / (100 - test_size), stratify=y_train_full, random_state=42)
        
        # Lưu vào session_state
        st.session_state.update({
            "X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_train": y_train, "y_val": y_val, "y_test": y_test
        })
        
        summary_df = pd.DataFrame({"Tập dữ liệu": ["Train", "Validation", "Test"], "Số lượng mẫu": [X_train.shape[0], X_val.shape[0], X_test.shape[0]]})
        st.success("✅ Dữ liệu đã được chia thành công!")
        st.table(summary_df)

def thi_nghiem():
    st.title("🧠 Huấn luyện Neural Network trên MNIST")
    
    if "X_train" not in st.session_state:
        st.error("⚠️ Chưa có dữ liệu! Hãy chia dữ liệu trước.")
        return
    
    X_train, X_val, X_test = [st.session_state[k].reshape(-1, 28 * 28) / 255.0 for k in ["X_train", "X_val", "X_test"]]
    y_train, y_val, y_test = [st.session_state[k] for k in ["y_train", "y_val", "y_test"]]
    
    k_folds = st.slider("Số fold cho Cross-Validation:", 3, 10, 5)
    num_layers = st.slider("Số lớp ẩn:", 1, 5, 2)
    num_neurons = st.slider("Số neuron mỗi lớp:", 32, 512, 128, 32)
    activation = st.selectbox("Hàm kích hoạt:", ["relu", "sigmoid", "tanh"])
    optimizer = st.selectbox("Optimizer:", ["adam", "sgd", "rmsprop"])
    loss_fn = "sparse_categorical_crossentropy"
    run_name = st.text_input("🔹 Nhập tên Run:", "Default_Run")
    
    if st.button("🚀 Huấn luyện mô hình"):
        with st.spinner("Đang huấn luyện..."):
            mlflow.start_run(run_name=run_name)
            mlflow.log_params({"num_layers": num_layers, "num_neurons": num_neurons, "activation": activation, "optimizer": optimizer, "k_folds": k_folds})
            
            kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
            accuracies, losses = [], []
            
            for train_idx, val_idx in kf.split(X_train, y_train):
                X_k_train, X_k_val = X_train[train_idx], X_train[val_idx]
                y_k_train, y_k_val = y_train[train_idx], y_train[val_idx]
                
                model = keras.Sequential([layers.Input(shape=(X_k_train.shape[1],))] + [layers.Dense(num_neurons, activation=activation) for _ in range(num_layers)] + [layers.Dense(10, activation="softmax")])
                model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
                
                start_time = time.time()
                history = model.fit(X_k_train, y_k_train, epochs=20, validation_data=(X_k_val, y_k_val), verbose=0)
                elapsed_time = time.time() - start_time
                
                accuracies.append(history.history["val_accuracy"][-1])
                losses.append(history.history["val_loss"][-1])
                
            avg_val_accuracy = np.mean(accuracies)
            avg_val_loss = np.mean(losses)
            
            mlflow.log_metrics({"avg_val_accuracy": avg_val_accuracy, "avg_val_loss": avg_val_loss, "elapsed_time": elapsed_time})
            
            test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
            mlflow.log_metrics({"test_accuracy": test_accuracy, "test_loss": test_loss})
            mlflow.end_run()
            
            st.success(f"✅ Huấn luyện hoàn tất!")
            st.write(f"📊 **Độ chính xác trung bình trên tập validation:** {avg_val_accuracy:.4f}")
            st.write(f"📊 **Độ chính xác trên tập test:** {test_accuracy:.4f}")


            
            
            

import streamlit as st
import numpy as np
import joblib
import random
import pandas as pd
import time
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model

def preprocess_canvas_image(canvas_result):
    """Chuyển đổi ảnh từ canvas sang định dạng phù hợp để dự đoán."""
    if canvas_result.image_data is None:
        return None
    img = canvas_result.image_data[:, :, :3]  # Chỉ lấy 3 kênh RGB
    img = Image.fromarray(img).convert("L").resize((28, 28))  # Chuyển sang grayscale, resize về 28x28
    img = np.array(img) / 255.0  # Chuẩn hóa về [0,1]
    img = img.reshape(1, -1)  # Đưa về dạng vector giống như trong `thi_nghiem()`
    return img

def du_doan():
    st.header("✍️ Vẽ số để dự đoán")

    # 📥 Load mô hình đã huấn luyện
    if "trained_model" in st.session_state:
        model = st.session_state["trained_model"]
        st.success("✅ Đã sử dụng mô hình vừa huấn luyện!")
    else:
        st.error("⚠️ Chưa có mô hình! Hãy huấn luyện trước.")


    # 🆕 Cập nhật key cho canvas khi nhấn "Tải lại"
    if "key_value" not in st.session_state:
        st.session_state.key_value = str(random.randint(0, 1000000))  

    if st.button("🔄 Tải lại nếu không thấy canvas"):
        st.session_state.key_value = str(random.randint(0, 1000000))  

    # ✍️ Vẽ số
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

    if st.button("Dự đoán số"):
        img = preprocess_canvas_image(canvas_result)

        if img is not None:
            st.image(Image.fromarray((img.reshape(28, 28) * 255).astype(np.uint8)), caption="Ảnh sau xử lý", width=100)

            # Dự đoán số
            prediction = model.predict(img)
            predicted_number = np.argmax(prediction, axis=1)[0]
            max_confidence = np.max(prediction)

            st.subheader(f"🔢 Dự đoán: {predicted_number}")
            st.write(f"📊 Mức độ tin cậy: {max_confidence:.2%}")

            # Hiển thị bảng confidence score
            prob_df = pd.DataFrame(prediction.reshape(1, -1), columns=[str(i) for i in range(10)]).T
            prob_df.columns = ["Mức độ tin cậy"]
            st.bar_chart(prob_df)

        else:
            st.error("⚠️ Hãy vẽ một số trước khi bấm Dự đoán!")

    
from datetime import datetime    
import streamlit as st
import mlflow
from datetime import datetime

def show_experiment_selector():
    st.title("📊 MLflow")
    
    # Kết nối với DAGsHub MLflow Tracking
    mlflow.set_tracking_uri("https://dagshub.com/Phuocbinh2003/Hoc_may_python.mlflow")
    
    # Lấy danh sách tất cả experiments
    experiment_name = "NN"
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
    
    # Lấy danh sách run_name từ params
    run_info = []
    for _, run in runs.iterrows():
        run_id = run["run_id"]
        run_params = mlflow.get_run(run_id).data.params
        run_name = run_params.get("run_name", f"Run {run_id[:8]}")
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
    
    
    
    tab1, tab2, tab3,tab4 = st.tabs(["📘 Lý thuyết TRAINING NEURAL NETWORK", "📘 TRAINING NEURAL NETWORK", "DEMO","🔥 Mlflow"] )

    with tab1:
        explain_nn()

    with tab2:
        split_data()
        thi_nghiem()
    
    with tab3:
        du_doan()
    with tab4:
        show_experiment_selector()


if __name__ == "__main__":
    pca_tsne()