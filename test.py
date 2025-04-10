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

    
import streamlit as st

def show_prediction_table():
    st.table({
        "Ảnh": ["Ảnh 1", "Ảnh 2", "Ảnh 3", "Ảnh 4", "Ảnh 5"],
        "Dự đoán": [7, 2, 3, 5, 8],
        "Xác suất": [0.98, 0.85, 0.96, 0.88, 0.97],
        "Gán nhãn?": ["✅", "❌", "✅", "❌", "✅"]
    })

def explain_Pseudo_Labelling():
    
    
    st.markdown("## 📚 Lý thuyết về Pseudo Labelling")
    st.write("""
    **Pseudo Labelling** là một phương pháp semi-supervised learning giúp kết hợp dữ liệu có nhãn và không nhãn để cải thiện độ chính xác của mô hình. Quá trình này diễn ra qua các bước sau:
    
    1️⃣ **Huấn luyện mô hình ban đầu** trên một tập dữ liệu nhỏ (~1% tổng số dữ liệu có nhãn).  
    2️⃣ **Dự đoán nhãn** cho các mẫu chưa được gán nhãn bằng mô hình đã huấn luyện.  
    3️⃣ **Lọc các dự đoán có độ tin cậy cao** dựa trên ngưỡng xác suất (ví dụ: > 0.95).  
    4️⃣ **Gán nhãn giả (Pseudo Labels)** cho các mẫu tin cậy.  
    5️⃣ **Thêm dữ liệu đã gán nhãn giả vào tập train**, mở rộng dữ liệu huấn luyện.  
    6️⃣ **Huấn luyện lại mô hình** với tập dữ liệu mở rộng (gồm dữ liệu ban đầu + dữ liệu có nhãn giả).  
    7️⃣ **Lặp lại các bước trên** cho đến khi đạt điều kiện dừng (hội tụ hoặc số lần lặp tối đa).  
    """)

    st.image("buoi8/img1.png", caption="Các bước Pseudo Labelling", use_container_width ="auto")

    # Ví dụ minh họa
    st.markdown("## 🔍 Ví dụ về Pseudo Labelling")
    st.write("""
    Giả sử ta có 70.000 ảnh chữ số viết tay (0-9), nhưng chỉ có 1% (100 ảnh) với mỗi số được gán nhãn ban đầu.  
    → Còn lại 60.000 ảnh không nhãn.
    """)

    st.markdown("### 🏁 **Bước 1: Huấn luyện mô hình ban đầu**")
    st.write("""
    - Mô hình được train trên 1000 ảnh có nhãn.  
    - Do dữ liệu ít, mô hình có độ chính xác thấp.  
    """)

    st.markdown("### 🧠 **Bước 2: Dự đoán nhãn cho dữ liệu chưa gán nhãn**")
    st.write("""
    - Chạy mô hình trên 60.000 ảnh chưa gán nhãn.  
    - Dự đoán và tính xác suất cho mỗi ảnh.  
    """)
    
    show_prediction_table()  # Hiển thị bảng dự đoán mẫu

    st.markdown("### 🔬 **Bước 3: Lọc dữ liệu có độ tin cậy cao**")
    st.write("""
    - Chỉ chọn những ảnh có xác suất dự đoán cao hơn ngưỡng tin cậy (ví dụ: 0.95).  
    - Ảnh 1, 3, 5 sẽ được gán nhãn giả.
    - Ảnh 2, 4 bị bỏ qua vì mô hình không tự tin.
    - Những ảnh đạt tiêu chuẩn sẽ được gán nhãn giả.  
    - Ảnh có độ tin cậy thấp sẽ bị loại bỏ.  
    """)

    st.markdown("### 🏷️ **Bước 4: Gán nhãn giả cho các dự đoán tin cậy**")
    st.write("""
    - Các mẫu có độ tin cậy cao được gán nhãn theo kết quả dự đoán của mô hình.  
    - ví dụ có 500 ảnh được gán nhãn giả.
    """)

    st.markdown("### 📂 **Bước 5: Thêm dữ liệu gán nhãn giả vào tập train**")
    st.write("""
    - Tập train mới = dữ liệu ban đầu + các ảnh có nhãn giả.  
    - Ví dụ: từ 1000 ảnh có nhãn ban đầu, ta có thể mở rộng lên 1500 ảnh sau khi thêm nhãn giả.  
    """)

    st.markdown("### 🔄 **Bước 6: Huấn luyện lại mô hình với tập dữ liệu mở rộng**")
    st.write("""
    - Huấn luyện lại mô hình trên tập dữ liệu mới.  
    - Mô hình sẽ học thêm từ dữ liệu mới và dần cải thiện độ chính xác.  
    """)

    st.markdown("### 🔁 **Bước 7: Lặp lại quá trình đến khi hội tụ**")
    st.write("""
    - Quá trình tiếp tục cho đến khi đạt điều kiện dừng:  
      - Đạt số lần lặp tối đa  
      - Mô hình không cải thiện thêm  
    """)

    st.markdown("## 🎯 **Kết quả cuối cùng**")
    st.write("""
    - Ban đầu chỉ có 100 ảnh có nhãn.  
    - Sau vài vòng lặp, mô hình có thể tự gán nhãn cho hàng ngàn ảnh.  
    - Độ chính xác tăng dần theo mỗi lần huấn luyện lại.  
    """)

    # st.success("✅ Pseudo Labelling giúp tận dụng dữ liệu chưa có nhãn để cải thiện mô hình AI một cách hiệu quả!")






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
    num = 0
    if "X_train" not in st.session_state:
        st.error("⚠️ Chưa có dữ liệu! Hãy chia dữ liệu trước.")
        return

    X_train, X_val, X_test = [st.session_state[k].reshape(-1, 28 * 28) / 255.0 for k in ["X_train", "X_val", "X_test"]]
    y_train, y_val, y_test = [st.session_state[k] for k in ["y_train", "y_val", "y_test"]]
    st.title(f"Chọn tham số cho Neural Network ")
    k_folds = st.slider("Số fold cho Cross-Validation:", 3, 10, 5)
    num_layers = st.slider("Số lớp ẩn:", 1, 5, 2)
    neurons_per_layer = []
    for i in range(num_layers):
        neurons_per_layer.append(st.slider(f"Số neuron lớp {i+1}:", 32, 512, 128, 32))
    activation = st.selectbox("Hàm kích hoạt:", ["relu", "sigmoid", "tanh"])
    optimizer = st.selectbox("Optimizer:", ["adam", "sgd", "rmsprop"])
    epochs = st.slider("🕰 Số epochs:", min_value=1, max_value=50, value=20, step=1)
    learning_rate = st.slider("⚡ Tốc độ học (Learning Rate):", min_value=1e-5, max_value=1e-1, value=1e-3, step=1e-5, format="%.5f")
    
    st.title(f"Chọn tham số cho Pseudo Labelling ")
    labeled_ratio = st.slider("📊 Tỉ lệ dữ liệu có nhãn ban đầu (%):", min_value=1, max_value=20, value=1, step=1)
    max_iterations = st.slider("🔄 Số lần lặp tối đa của Pseudo-Labeling:", min_value=1, max_value=10, value=3, step=1)
    confidence_threshold = st.slider("✅ Ngưỡng tin cậy Pseudo Labeling (%):", min_value=50, max_value=99, value=95, step=1) / 100.0

    loss_fn = "sparse_categorical_crossentropy"
    run_name = st.text_input("🔹 Nhập tên Run:", "Default_Run")
    st.session_state['run_name'] = run_name

    if st.button("🚀 Huấn luyện mô hình"):
        with st.spinner("Đang huấn luyện..."):
            mlflow.start_run(run_name=run_name)
            mlflow.log_params({
                "num_layers": num_layers,
                "num_neurons": neurons_per_layer,
                "activation": activation,
                "optimizer": optimizer,
                "learning_rate": learning_rate,
                "k_folds": k_folds,
                "epochs": epochs,
                "labeled_ratio": labeled_ratio,
                "max_iterations": max_iterations,
                "confidence_threshold": confidence_threshold
            })

            num_labeled = int(len(X_train) * labeled_ratio / 100)
            labeled_idx = np.random.choice(len(X_train), num_labeled, replace=False)
            unlabeled_idx = np.setdiff1d(np.arange(len(X_train)), labeled_idx)

            X_labeled, y_labeled = X_train[labeled_idx], y_train[labeled_idx]
            X_unlabeled = X_train[unlabeled_idx]

            total_pseudo_labels = 0  # Tổng số nhãn giả được thêm vào

            for iteration in range(max_iterations):
                kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
                accuracies, losses = [], []
                training_progress = st.progress(0)
                training_status = st.empty()

                num = 0
                total_steps = k_folds * max_iterations

                for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_labeled, y_labeled)):
                    X_k_train, X_k_val = X_labeled[train_idx], X_labeled[val_idx]
                    y_k_train, y_k_val = y_labeled[train_idx], y_labeled[val_idx]

                    model = keras.Sequential([
                        layers.Input(shape=(X_k_train.shape[1],))
                    ] + [
                        layers.Dense(neurons_per_layer[i], activation=activation) for i in range(num_layers)
                    ] + [
                        layers.Dense(10, activation="softmax")
                    ])

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
                    num += 1
                    progress_percent = int((num / k_folds) * 100)

                    training_progress.progress(progress_percent)
                    training_status.text(f"⏳ Đang huấn luyện... {progress_percent}%")

                avg_val_accuracy = np.mean(accuracies)
                avg_val_loss = np.mean(losses)

                mlflow.log_metrics({
                    "avg_val_accuracy": avg_val_accuracy,
                    "avg_val_loss": avg_val_loss,
                    "elapsed_time": elapsed_time
                })

                pseudo_preds = model.predict(X_unlabeled)
                pseudo_labels = np.argmax(pseudo_preds, axis=1)
                confidence_scores = np.max(pseudo_preds, axis=1)
                confident_mask = confidence_scores > confidence_threshold

                num_pseudo_added = np.sum(confident_mask)
                total_pseudo_labels += num_pseudo_added
                
                
                
                # Lưu các mẫu pseudo-labels để visualize
                X_pseudo = X_unlabeled[confident_mask][:10]  # Lấy 10 mẫu có độ tin cậy cao nhất
                y_pseudo = pseudo_labels[confident_mask][:10]

                X_labeled = np.concatenate([X_labeled, X_unlabeled[confident_mask]])
                y_labeled = np.concatenate([y_labeled, pseudo_labels[confident_mask]])
                X_unlabeled = X_unlabeled[~confident_mask]

                # Đánh giá mô hình trên tập validation và test sau khi gán nhãn giả
                #val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
                test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
                st.write(f"Số lượng mẫu pseudo-label có độ tin cậy cao: {len(X_pseudo)}")
                
                if len(X_pseudo) > 0:
                    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
                    for i, ax in enumerate(axes.flat[:len(X_pseudo)]):
                        ax.imshow(X_pseudo[i].reshape(28, 28), cmap='gray')
                        ax.set_title(f"Label: {y_pseudo[i]}")
                        ax.axis("off")
                    st.pyplot(fig)
                else:
                    st.warning("⚠️ Không có mẫu pseudo-label nào đạt ngưỡng tin cậy để hiển thị.")
                
                st.write(f"Số lượng dữ liệu chưa gán nhãn còn lại sau vòng {iteration+1}: {len(X_unlabeled)}")

                st.write(f"📢 **Vòng lặp {iteration+1}:**")
                st.write(f"- Số pseudo labels mới thêm: {num_pseudo_added}")
                st.write(f"- Tổng số pseudo labels: {total_pseudo_labels}")
                
                # st.write(f"- 🔥 **Độ chính xác trên tập validation:** {val_accuracy:.4f}")
                st.write(f"- 🚀 **Độ chính xác trên tập test:** {test_accuracy:.4f}")
                st.write("---")

                # Lưu độ chính xác vào MLflow để theo dõi
                mlflow.log_metrics({
                    # f"val_accuracy_iter_{iteration+1}": val_accuracy,
                    f"test_accuracy_iter_{iteration+1}": test_accuracy
                })
                if len(X_unlabeled) == 0:
                    break
            
            # Sau khi hoàn thành Pseudo Labeling, huấn luyện lại mô hình với dữ liệu đã gán nhãn
            st.write("🔄 **Huấn luyện lại mô hình với toàn bộ dữ liệu đã gán nhãn**...")

            model_final = keras.Sequential([
                layers.Input(shape=(X_labeled.shape[1],))
            ] + [
                layers.Dense(neurons_per_layer[i], activation=activation) for i in range(num_layers)
            ] + [
                layers.Dense(10, activation="softmax")
            ])

            if optimizer == "adam":
                opt = keras.optimizers.Adam(learning_rate=learning_rate)
            elif optimizer == "sgd":
                opt = keras.optimizers.SGD(learning_rate=learning_rate)
            else:
                opt = keras.optimizers.RMSprop(learning_rate=learning_rate)

            model_final.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"])

            with st.spinner("🔁 Đang huấn luyện lại mô hình..."):
                history_final = model_final.fit(X_labeled, y_labeled, epochs=epochs, validation_data=(X_val, y_val), verbose=0)

            final_test_loss, final_test_accuracy = model_final.evaluate(X_test, y_test, verbose=0)

            # Log kết quả sau huấn luyện lại
            mlflow.log_metrics({
                "final_test_accuracy": final_test_accuracy,
                "final_test_loss": final_test_loss
            })

            
            st.write(f"📊 **Độ chính xác cuối cùng trên tập test:** {final_test_accuracy:.4f}")

            # Lưu mô hình đã huấn luyện lại vào session_state
            # ====== SAU KHI TRAIN XONG ======
            run_name = st.session_state.get('run_name')
            if not run_name:
                st.error("⚠️ Tên run không tồn tại!")
                return

            model_key = f"trained_model_{run_name}"

            # Kiểm tra model hợp lệ trước khi lưu
            try:
                model_final.predict(X_test[:1])  # Kiểm tra tính hợp lệ của model
                if model_key in st.session_state:
                    st.warning(f"⚠️ Model `{model_key}` đã tồn tại và sẽ bị ghi đè!")

                st.session_state[model_key] = model_final
                st.success(f"✅ Đã lưu model thành công với key: `{model_key}`")

                # Debug: Hiển thị tất cả keys (chỉ hiển thị nếu cần)
                st.write("📌 Các keys trong session state:", list(st.session_state.keys()))

            except Exception as e:
                st.exception(e)

            st.success(f"✅ Mô hình cuối cùng đã được lưu vào session_state với tên `{st.session_state['run_name']}`!")
            
            
            test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
            mlflow.log_metrics({"test_accuracy": test_accuracy, "test_loss": test_loss})
            mlflow.end_run()
            #st.session_state[f"trained_model_{st.session_state['run_name']}"] = model

            training_progress.progress(100)
            training_status.text("✅ Huấn luyện hoàn tất!")

            st.success(f"✅ Huấn luyện hoàn tất!")
            st.write(f"📊 **Độ chính xác trung bình trên tập validation:** {avg_val_accuracy:.4f}")
            st.write(f"📊 **Độ chính xác trên tập test:** {test_accuracy:.4f}")
            st.success(f"✅ Đã log dữ liệu cho **{st.session_state['run_name']}** trong MLflow! 🚀")
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

 
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(experiment_ids=['9'], order_by=["start_time DESC"], max_results=5)
    
    # Tạo dictionary ánh xạ tên mô hình sang run_i
    model_dict = {run.data.tags.get("mlflow.runName", "Unknown"): run.info.run_id for run in runs}

    if not model_dict:
        st.error("⚠️ Chưa có mô hình nào! Hãy huấn luyện trước.")
        return

    # 🔍 Dropdown chọn model theo tên
    selected_model = st.selectbox(
        "🔍 Chọn mô hình đã train:",
        options=list(model_dict.keys())
    )

    # 🚨 Nút tải model
    if st.button("⬇️ Tải model"):
        try:
            with st.spinner("Đang tải model..."):
                model_uri = f"runs:/{model_dict[selected_model]}/model"
                st.session_state.model = mlflow.keras.load_model(model_uri)
                st.session_state.model_loaded = True
                st.success(f"✅ Đã tải thành công model: {selected_model}")
        except Exception as e:
            st.error(f"❌ Lỗi khi tải model: {str(e)}")
            return

    # Chỉ hiển thị canvas khi model đã được load
    if 'model_loaded' not in st.session_state:
        st.info("👉 Vui lòng chọn model và nhấn nút [Tải model] trước")
        return

    # 🎨 Khởi tạo canvas key
    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = 0

    # 🔄 Nút reset canvas
    if st.button("🔄 Tạo canvas mới"):
        st.session_state.canvas_key += 1

    # ✍️ Vùng vẽ số
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        height=200,
        width=200,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}",
        update_streamlit=True
    )

    # 🎯 Nút dự đoán
    if st.button("🔮 Dự đoán"):
        if canvas_result.image_data is not None:
            # Tiền xử lý ảnh
            img = preprocess_canvas_image(canvas_result)
            
            # Hiển thị ảnh đã xử lý
            st.image(Image.fromarray((img.reshape(28, 28) * 255).astype(np.uint8), 
                    caption="Ảnh đã xử lý", width=150))

            # Dự đoán
            prediction = st.session_state.model.predict(img)
            predicted_num = np.argmax(prediction)
            confidence = np.max(prediction)

            # Hiển thị kết quả
            st.subheader(f"📊 Kết quả: {predicted_num}")
            st.metric(label="Độ tin cậy", value=f"{confidence:.2%}")

            # Biểu đồ xác suất
            prob_df = pd.DataFrame({
                'Số': range(10),
                'Xác suất': prediction[0]
            })
            st.bar_chart(prob_df, x='Số', y='Xác suất')
            
        else:
            st.warning("⚠️ Vui lòng vẽ số vào canvas trước khi dự đoán")

from datetime import datetime   
def show_experiment_selector():
    st.title("📊 MLflow Experiments - DAGsHub")

    # Kết nối với DAGsHub MLflow Tracking
    
    # Lấy danh sách tất cả experiments
    experiment_name = "Classification"
    
    # Tìm experiment theo tên
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
        start_time_ms = selected_run.info.start_time  # Thời gian lưu dưới dạng millisecondT

# Chuyển sang định dạng ngày giờ dễ đọc
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
        # dataset_path = f"{selected_experiment.artifact_location}/{selected_run_id}/artifacts/dataset.csv"
        # st.write("### 📂 Dataset:")
        # st.write(f"📥 [Tải dataset]({dataset_path})")
    else:
        st.warning("⚠ Không tìm thấy thông tin cho run này.")
           
                  
        
        
          
import mlflow
import os
from mlflow.tracking import MlflowClient
def Semi_supervised():
    #st.title("🚀 MLflow DAGsHub Tracking với Streamlit")
    
    if "mlflow_initialized" not in st.session_state:   
        DAGSHUB_MLFLOW_URI = "https://dagshub.com/Phuocbinh2003/Hoc_may_python.mlflow"
        st.session_state['mlflow_url']=DAGSHUB_MLFLOW_URI
        mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)

        os.environ["MLFLOW_TRACKING_USERNAME"] = "Phuocbinh2003"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "c1495823c8f9156923b06f15899e989db7e62052"
        st.session_state.mlflow_initialized = True
        mlflow.set_experiment("Semi_supervised")   
        
    
    
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
        explain_Pseudo_Labelling()

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
    Semi_supervised()
