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
    
    st.image("buoi7/img1.webp", caption="Cấu trúc mạng nơ-ron(medium.com)", use_container_width="auto")
    
   

    st.markdown("""
    ## 📌 Công thức toán học trong Neural Network:
    Mỗi nơ-ron trong một lớp nhận tín hiệu từ các nơ-ron lớp trước, nhân với trọng số (**weights**), cộng với **bias**, rồi đưa vào một **hàm kích hoạt** để quyết định tín hiệu truyền đi.
    """)

    st.markdown("### 🎯 Công thức tính giá trị đầu ra của một nơ-ron:")
    st.latex(r" z = \sum_{i=1}^{n} w_i x_i + b ")

    st.markdown(r"""
    Trong đó:
    - $$ x_i $$ là đầu vào (**input features**).
    - $$ w_i $$ là **trọng số** (**weights**) kết nối với nơ-ron đó.
    - $$ b $$ là **bias** (hệ số dịch chuyển).
    - $$ z $$ là tổng có trọng số (**weighted sum**).

    Sau khi tính toán $$ z $$, nó sẽ đi qua một **hàm kích hoạt** $$ \sigma(z) $$ để tạo ra giá trị đầu ra.
    """)

    # st.markdown("### 🎯 Công thức tính đầu ra sau khi qua hàm kích hoạt:")
    # st.latex(r" a = \sigma(z) ")


    
    st.markdown("""
    ### 🎯 Hàm Kích Hoạt (Activation Functions)
    Hàm kích hoạt giúp mạng học được các tính phi tuyến tính, nhờ đó có thể mô hình hóa các mối quan hệ phức tạp.
    """)
    
    st.image("buoi7/img2.png", caption="Một số hàm kích hoạt cơ bản", use_container_width="auto")
    
    st.markdown("- **Sigmoid:** Chuyển đổi giá trị đầu vào thành khoảng từ 0 đến 1, phù hợp cho bài toán phân loại nhị phân.")
    st.latex(r"\sigma(z) = \frac{1}{1 + e^{-z}}")

    st.markdown("- **Tanh (Hyperbolic Tangent):** Đầu ra nằm trong khoảng từ -1 đến 1, giúp xử lý dữ liệu có cả giá trị dương và âm.")
    st.latex(r"\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}")

    st.markdown("- **ReLU (Rectified Linear Unit):** Nếu đầu vào âm thì bằng 0, còn nếu dương thì giữ nguyên giá trị.")
    st.latex(r"ReLU(z) = \max(0, z)")
    
   

    st.markdown("### 🔄 Quá trình huấn luyện Neural Network")
    st.markdown("Mạng nơ-ron học bằng cách cập nhật các trọng số thông qua hai giai đoạn chính:")

    st.markdown("#### 1️⃣ Lan truyền thuận (Forward Propagation)")
    st.markdown("- Input đi qua từng lớp nơ-ron, tính toán đầu ra:")
    st.latex(r"a^{(l)} = \sigma(W^{(l)} a^{(l-1)} + b^{(l)})")

    st.markdown("Trong đó:")
    st.markdown(r"- $$ a^{(l)} $$: Đầu ra của lớp thứ $$l $$.")
    st.markdown(r"- $$ W^{(l)} $$: Ma trận trọng số giữa lớp $$l-1 $$ và lớp $$ l $$.")
    st.markdown(r"- $$ a^{(l-1)} $$: Đầu ra của lớp trước đó (hoặc là đầu vào nếu $$ l = 1 $$).")
    st.markdown(r"- $$b^{(l)} $$: Bias của lớp $$ l $$.")
    st.markdown(r"- $$ \sigma(z) $$: Hàm kích hoạt (ReLU, Sigmoid, Tanh,...).")

    st.markdown("#### 2️⃣ Tính toán loss")
    st.markdown("- Hàm mất mát đo lường sai số giữa dự đoán và thực tế.")
    st.markdown("- Ví dụ: Mean Squared Error (MSE) cho bài toán hồi quy:")
    st.latex(r"L = \frac{1}{N} \sum (y_{true} - y_{pred})^2")

    st.markdown("- Cross-Entropy Loss cho bài toán phân loại:")
    st.latex(r"L = - \sum y_{true} \log(y_{pred})")

    st.markdown("Trong đó:")
    st.markdown(r"- $$ L $$: Giá trị hàm mất mát.")
    st.markdown(r"- $$ N $$: Số lượng mẫu trong tập dữ liệu.")
    st.markdown(r"- $$y_{true} $$: Nhãn thực tế của dữ liệu.")
    st.markdown(r"- $$y_{pred} $$: Dự đoán của mô hình.")

    st.markdown("#### 3️⃣ Lan truyền ngược (Backpropagation)")
    st.markdown("- Tính đạo hàm của hàm mất mát theo trọng số.")
    st.markdown("- Sử dụng thuật toán tối ưu để cập nhật trọng số.")

    st.markdown("Lan truyền ngược dựa trên công thức:")
    st.latex(r"\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial a^{(l)}} \cdot \frac{\partial a^{(l)}}{\partial z^{(l)}} \cdot \frac{\partial z^{(l)}}{\partial W^{(l)}}")

    st.markdown("Trong đó:")
    st.markdown(r"- $$\frac{\partial L}{\partial W^{(l)}} $$: Đạo hàm của loss theo trọng số $$ W^{(l)} $$.")
    st.markdown(r"- $$ \frac{\partial L}{\partial a^{(l)}} $$: Đạo hàm của loss theo đầu ra của lớp \( l \).")
    st.markdown(r"- $$ \frac{\partial a^{(l)}}{\partial z^{(l)}} $$: Đạo hàm của hàm kích hoạt.")
    st.markdown(r"- $$ \frac{\partial z^{(l)}}{\partial W^{(l)}} $$: Đạo hàm của đầu vào trước lớp kích hoạt theo trọng số.")

    st.markdown("#### 4️⃣ Tối ưu hóa")
    st.markdown("- **Gradient Descent:** Cập nhật trọng số bằng cách đi theo hướng giảm của gradient.")
    st.latex(r"W^{(l)} = W^{(l)} - \alpha \frac{\partial L}{\partial W^{(l)}}")
    st.markdown("- **Momentum:** Thêm động lượng giúp tối ưu nhanh hơn.")
    st.latex(r"v_t = \beta v_{t-1} + (1 - \beta) \frac{\partial L}{\partial W^{(l)}}")
    st.latex(r"W^{(l)} = W^{(l)} - \alpha v_t")
    st.markdown("- **Adam (Adaptive Moment Estimation):** Kết hợp Momentum và RMSprop để đạt hiệu suất tối ưu.")
    st.latex(r"m_t = \beta_1 m_{t-1} + (1 - \beta_1) \frac{\partial L}{\partial W^{(l)}}")
    st.latex(r"v_t = \beta_2 v_{t-1} + (1 - \beta_2) \left(\frac{\partial L}{\partial W^{(l)}}\right)^2")
    st.latex(r"\hat{m_t} = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v_t} = \frac{v_t}{1 - \beta_2^t}")
    st.latex(r"W^{(l)} = W^{(l)} - \alpha \frac{\hat{m_t}}{\sqrt{\hat{v_t}} + \epsilon}")

    st.markdown("Trong đó:")
    st.markdown(r"- $$ \alpha $$: Tốc độ học (learning rate).")
    st.markdown(r"- $$ v_t $$: Giá trị động lượng tại thời điểm $$ t $$.")
    st.markdown(r"- $$ \beta, \beta_1, \beta_2 $$: Hệ số Momentum hoặc Adam.")
    st.markdown(r"- $$ m_t $$, $$ v_t $$: Trung bình trọng số và phương sai của gradient.")
    st.markdown(r"- $$ \epsilon $$: Số rất nhỏ để tránh chia cho 0.")

    st.markdown("""
    ### 🔍 Kết Luận
    Neural Network là một mô hình mạnh mẽ trong Machine Learning và Deep Learning, có thể học được các đặc trưng phức tạp từ dữ liệu. Hiểu rõ cách hoạt động giúp ta tối ưu hóa mô hình để đạt hiệu suất cao hơn.
    """)
    
    
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
    st.image("buoi4/img3.png", caption="Một số hình ảnh từ MNIST Dataset", use_container_width ="auto")

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

    st.subheader("📊 Minh họa dữ liệu MNIST")

    # Đọc và hiển thị ảnh GIF minh họa
    gif_path = "buoi7/g1.gif"  # Thay bằng đường dẫn thực tế
    st.image(gif_path, caption="Hình ảnh minh họa dữ liệu MNIST", use_container_width="auto")

    # Mô tả về dữ liệu MNIST
    st.write("""
    Dữ liệu MNIST bao gồm các hình ảnh chữ số viết tay có kích thước **28x28 pixels**.  
    Mỗi ảnh là một **ma trận 28x28**, với mỗi pixel có giá trị từ **0 đến 255**.  
    Khi đưa vào mô hình, ảnh sẽ được biến đổi thành **784 features (28x28)** để làm đầu vào cho mạng nơ-ron.  
    Mô hình sử dụng các lớp ẩn để học và dự đoán chính xác chữ số từ hình ảnh.
    """)

    
    



# ======================================

# def input_mlflow():
#     DAGSHUB_MLFLOW_URI = "https://dagshub.com/Phuocbinh2003/Hoc_may_python.mlflow"
#     mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
#     st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI
#     os.environ["MLFLOW_TRACKING_USERNAME"] = "Phuocbinh2003"
#     os.environ["MLFLOW_TRACKING_PASSWORD"] = "c1495823c8f9156923b06f15899e989db7e62052"
#     mlflow.set_experiment("NN")

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
    num_samples = st.slider("📌 Chọn số lượng ảnh để huấn luyện:", 1000, total_samples, 10000)
    num_samples =num_samples -10
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
    epochs = st.slider("🕰 Số epochs:", min_value=1, max_value=50, value=20, step=1)
    learning_rate = st.slider("⚡ Tốc độ học (Learning Rate):", min_value=1e-5, max_value=1e-1, value=1e-3, step=1e-5, format="%.5f")

    loss_fn = "sparse_categorical_crossentropy"
    run_name = st.text_input("🔹 Nhập tên Run:", "Default_Run")
    st.session_state['run_name'] = run_name
    
    if st.button("🚀 Huấn luyện mô hình"):
        with st.spinner("Đang huấn luyện..."):
            mlflow.start_run(run_name=run_name)
            mlflow.log_params({
                "num_layers": num_layers,
                "num_neurons": num_neurons,
                "activation": activation,
                "optimizer": optimizer,
                "learning_rate": learning_rate,
                "k_folds": k_folds,
                "epochs": epochs
            })

            kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
            accuracies, losses = [], []

            # Thanh tiến trình tổng quát cho toàn bộ quá trình huấn luyện
            training_progress = st.progress(0)
            training_status = st.empty()

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
                X_k_train, X_k_val = X_train[train_idx], X_train[val_idx]
                y_k_train, y_k_val = y_train[train_idx], y_train[val_idx]

                model = keras.Sequential([
                    layers.Input(shape=(X_k_train.shape[1],))
                ] + [
                    layers.Dense(num_neurons, activation=activation) for _ in range(num_layers)
                ] + [
                    layers.Dense(10, activation="softmax")
                ])

                # Chọn optimizer với learning rate
                if optimizer == "adam":
                    opt = keras.optimizers.Adam(learning_rate=learning_rate)
                elif optimizer == "sgd":
                    opt = keras.optimizers.SGD(learning_rate=learning_rate)
                else:
                    opt = keras.optimizers.RMSprop(learning_rate=learning_rate)

                model.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"])

                start_time = time.time()
                history = model.fit(X_k_train, y_k_train, epochs=epochs, validation_data=(X_k_val, y_k_val), verbose=0)

                elapsed_time = time.time() - start_time
                accuracies.append(history.history["val_accuracy"][-1])
                losses.append(history.history["val_loss"][-1])

                # Cập nhật thanh tiến trình chính (theo fold)
                st.write(fold_idx ,k_folds)
                
                progress_percent = int((fold_idx + 1 / k_folds))
                training_progress.progress(progress_percent)
                
                            

                
                training_status.text(f"⏳ Đang huấn luyện... {progress_percent*100}%")

            avg_val_accuracy = np.mean(accuracies)
            avg_val_loss = np.mean(losses)

            mlflow.log_metrics({
                "avg_val_accuracy": avg_val_accuracy,
                "avg_val_loss": avg_val_loss,
                "elapsed_time": elapsed_time
            })

            test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
            mlflow.log_metrics({"test_accuracy": test_accuracy, "test_loss": test_loss})

            mlflow.end_run()
            st.session_state["trained_model"] = model

            # Hoàn thành tiến trình
            training_progress.progress(1.0)
            training_status.text("✅ Huấn luyện hoàn tất!")

            st.success(f"✅ Huấn luyện hoàn tất!")
            st.write(f"📊 **Độ chính xác trung bình trên tập validation:** {avg_val_accuracy:.4f}")
            st.write(f"📊 **Độ chính xác trên tập test:** {test_accuracy:.4f}")
            st.success(f"✅ Đã log dữ liệu cho **{st.session_state['run_name']}** trong MLflow (Neural_Network)! 🚀")
            st.markdown(f"🔗 [Truy cập MLflow UI]({st.session_state['mlflow_url']})")


                
            
            

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
    experiment_name = "Neural_Network"
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
def Neural_Network():
    #st.title("🚀 MLflow DAGsHub Tracking với Streamlit")
    
    if "mlflow_initialized" not in st.session_state:   
        DAGSHUB_MLFLOW_URI = "https://dagshub.com/Phuocbinh2003/Hoc_may_python.mlflow"
        st.session_state['mlflow_url']=DAGSHUB_MLFLOW_URI
        mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)

        os.environ["MLFLOW_TRACKING_USERNAME"] = "Phuocbinh2003"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "c1495823c8f9156923b06f15899e989db7e62052"
        st.session_state.mlflow_initialized = True
        mlflow.set_experiment("Neural_Network")   
        
    
    
    # Tạo các tab với tiêu đề tương ứng
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📘 Lý thuyết NEURAL NETWORK",
        "📊 Mẫu dữ liệu",
        "🧠 Huấn luyện",
        "🖥️ DEMO",
        "🔥 MLflow"
    ])

    # Nội dung từng tab
    with tab1:
        explain_nn()

    with tab2:
        data()

    with tab3:
        st.title("🧠 Huấn luyện Neural Network trên MNIST")
        split_data()
        thi_nghiem()

    with tab4:
        du_doan()

    with tab5:
        show_experiment_selector()



if __name__ == "__main__":
    Neural_Network()