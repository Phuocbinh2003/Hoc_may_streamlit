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

    **Neural Network (Mạng nơ-ron)** là mô hình lấy cảm hứng từ hoạt động của não bộ, gồm nhiều lớp nơ-ron nhân tạo kết nối với nhau.

    ### 🔰 Kiến trúc cơ bản:
    """)
    st.image("https://www.researchgate.net/publication/372377465/figure/fig1/AS:11431281141736600@1686735021201/Architecture-of-an-Artificial-Neural-Network-ANN.png", 
             width=600)
    
    st.markdown("""
    ### 📌 Các thành phần chính:
    1. **Input Layer**: Lớp tiếp nhận dữ liệu đầu vào
    2. **Hidden Layers**: Các lớp xử lý ẩn
    3. **Output Layer**: Lớp đưa ra kết quả dự đoán

    ### 🎯 Hàm kích hoạt (Activation Functions):
    """)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("https://miro.medium.com/v2/resize:fit:1400/1*syRl4D2_FnIyVvy2ZCYCFQ.png", 
                 caption="Sigmoid", width=200)
    with col2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/Rectified_linear_unit.svg/1200px-Rectified_linear_unit.svg.png", 
                 caption="ReLU", width=200)
    with col3:
        st.image("https://paperswithcode.com/media/thumbnails/task/task-0000000607-aa5b1a4e.jpg", 
                 caption="Softmax", width=200)
    
    st.markdown("""
    ### 🔄 Quá trình lan truyền:
    1. **Lan truyền thuận (Forward Propagation)**: Tính toán đầu ra
    2. **Tính toán loss**: So sánh với giá trị thực
    3. **Lan truyền ngược (Backpropagation)**: Cập nhật trọng số
    4. **Tối ưu hóa**: Sử dụng các thuật toán như GD, Adam

    ### 📉 Hàm mất mát phổ biến:
    ```python
    Loss = CrossEntropy(y_true, y_pred)
    ```
    """)

# ======================================
# PHẦN 2: DEMO TRAINING NEURAL NETWORK
# ======================================
def load_mnist():
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X = mnist.data.astype('float32')
    y = mnist.target.astype('int64')
    return X, y

def create_model(layers, activations, dropout_rate):
    model = Sequential()
    for i, (units, activation) in enumerate(zip(layers, activations)):
        if i == 0:
            model.add(Dense(units, activation=activation, input_shape=(784,)))
        else:
            model.add(Dense(units, activation=activation))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='softmax'))
    return model

class MlflowLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({
            'train_loss': logs['loss'],
            'train_accuracy': logs['accuracy'],
            'val_loss': logs['val_loss'],
            'val_accuracy': logs['val_accuracy']
        }, step=epoch)

def input_mlflow():
    mlflow.set_tracking_uri("https://dagshub.com/Phuocbinh2003/Hoc_may_python.mlflow")
    os.environ["MLFLOW_TRACKING_USERNAME"] = "Phuocbinh2003"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "c1495823c8f9156923b06f15899e989db7e62052"
    mlflow.set_experiment("Neural_Network_MNIST")

def train_neural_net():
    st.title("🔢 Nhận diện chữ số viết tay với Neural Network")

    # Load dữ liệu
    X, y = load_mnist()
    
    # Chuẩn hóa dữ liệu
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Chia tập train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Sidebar controls
    st.sidebar.header("⚙️ Thiết lập tham số")
    layers = []
    activations = []
    
    num_layers = st.sidebar.slider("Số lớp ẩn", 1, 5, 2)
    for i in range(num_layers):
        col1, col2 = st.sidebar.columns(2)
        with col1:
            units = col1.number_input(f"Nơ-ron lớp {i+1}", 32, 512, 128, key=f"units_{i}")
        with col2:
            activation = col2.selectbox(
                f"Hàm kích hoạt lớp {i+1}",
                ["relu", "sigmoid", "tanh"],
                key=f"activation_{i}"
            )
        layers.append(units)
        activations.append(activation)
    
    dropout_rate = st.sidebar.slider("Dropout rate", 0.0, 0.5, 0.2)
    learning_rate = st.sidebar.selectbox("Learning rate", [1e-2, 1e-3, 1e-4], index=1)
    epochs = st.sidebar.slider("Epochs", 5, 50, 10)
    batch_size = st.sidebar.slider("Batch size", 32, 256, 128)

    # Tạo model
    model = create_model(layers, activations, dropout_rate)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # MLflow setup
    input_mlflow()
    run_name = st.text_input("Nhập tên run:", f"Run_{datetime.now().strftime('%Y%m%d%H%M')}")
    
    if st.button("🎬 Bắt đầu training"):
        with st.spinner("Đang training..."):
            with mlflow.start_run(run_name=run_name):
                # Log parameters
                mlflow.log_params({
                    "num_layers": num_layers,
                    "layers": layers,
                    "activations": activations,
                    "dropout_rate": dropout_rate,
                    "learning_rate": learning_rate,
                    "epochs": epochs,
                    "batch_size": batch_size
                })

                # Training
                start_time = time.time()
                history = model.fit(
                    X_train, y_train,
                    validation_split=0.2,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=0,
                    callbacks=[MlflowLogger()]
                )
                training_time = time.time() - start_time

                # Đánh giá
                test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
                
                # Log metrics
                mlflow.log_metrics({
                    "final_test_loss": test_loss,
                    "final_test_accuracy": test_acc,
                    "training_time": training_time
                })

                # Hiển thị kết quả
                st.subheader("📊 Kết quả training")
                col1, col2, col3 = st.columns(3)
                col1.metric("Train Accuracy", f"{history.history['accuracy'][-1]:.2%}")
                col2.metric("Validation Accuracy", f"{history.history['val_accuracy'][-1]:.2%}")
                col3.metric("Test Accuracy", f"{test_acc:.2%}")

                # Vẽ đồ thị
                fig = px.line(
                    history.history,
                    y=['loss', 'val_loss'],
                    labels={'value': 'Loss', 'variable': 'Loại'},
                    title='Biểu đồ Loss qua các epochs'
                )
                st.plotly_chart(fig)

                # Lưu model
                model.save("model.h5")
                mlflow.log_artifact("model.h5")
                st.success("✅ Training hoàn tất!")
                st.markdown(f"### 🔗 [Theo dõi trên MLflow UI]({mlflow.get_tracking_uri()})")

# ======================================
# PHẦN 3: GIAO DIỆN CHÍNH
# ======================================
def main():
    st.set_page_config(page_title="Neural Network MNIST", page_icon="🧠")
    
    tab1, tab2, tab3 = st.tabs(["📚 Lý thuyết", "🎮 Thử nghiệm", "📊 MLflow"])
    
    with tab1:
        explain_nn()
    
    with tab2:
        train_neural_net()
    
    with tab3:
        st.write("Truy cập MLflow để xem chi tiết các runs:")
        st.markdown("[🔗 MLflow Tracking trên DAGsHub](https://dagshub.com/Phuocbinh2003/Hoc_may_python.mlflow)")

if __name__ == "__main__":
    main()