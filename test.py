import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import plotly.express as px
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import os
# Hàm khởi tạo MLflow
def init_mlflow():
    mlflow.set_tracking_uri("https://dagshub.com/Phuocbinh2003/Hoc_may_python.mlflow")
    os.environ["MLFLOW_TRACKING_USERNAME"] = "Phuocbinh2003"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "c1495823c8f9156923b06f15899e989db7e62052"
    mlflow.set_experiment("Pseudo_Label_MNIST")

# Tab Lý thuyết
def theory_tab():
    st.markdown("""
    ## 📚 Lý thuyết Pseudo Labelling
    **Pseudo Labelling** là kỹ thuật semi-supervised learning kết hợp dữ liệu có nhãn và không nhãn:
    
    1. Huấn luyện model ban đầu trên tập nhãn nhỏ
    2. Dự đoán nhãn cho dữ liệu chưa gán nhãn
    3. Chọn các dự đoán có độ tin cậy cao làm pseudo labels
    4. Huấn luyện lại model với dữ liệu mở rộng
    5. Lặp lại quá trình cho đến khi hội tụ

    **Công thức chọn pseudo labels:**
    """)
    
    st.latex(r'''
    \text{Chọn mẫu } x_i \text{ nếu } \max(p(y|x_i)) \geq \tau
    ''')
    st.write("Trong đó τ là ngưỡng tin cậy (ví dụ: 0.95)")

# Tab Thí nghiệm
def experiment_tab():
    st.title("🔬 Thí nghiệm Pseudo Labelling")
    
    # Tải dữ liệu
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
    X = X / 255.0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Tham số")
        init_ratio = st.slider("Tỷ lệ dữ liệu ban đầu (%)", 1, 10, 1)
        threshold = st.slider("Ngưỡng tin cậy", 0.7, 0.99, 0.95)
        max_iter = st.slider("Số vòng lặp tối đa", 1, 20, 5)
        n_units = st.slider("Số nơ-ron lớp ẩn", 64, 512, 128)
        
    # Khởi tạo MLflow
    init_mlflow()
    run_name = st.text_input("Tên thí nghiệm:", "Pseudo_Label_Exp")
    
    if st.button("Bắt đầu thí nghiệm"):
        with st.spinner("Đang xử lý..."):
            with mlflow.start_run(run_name=run_name):
                # Log parameters
                mlflow.log_params({
                    "init_ratio": init_ratio,
                    "threshold": threshold,
                    "max_iter": max_iter,
                    "n_units": n_units
                })
                
                # Chọn dữ liệu ban đầu (1%)
                sss = StratifiedShuffleSplit(n_splits=1, test_size=1-init_ratio/100, random_state=42)
                for train_idx, _ in sss.split(X_train, y_train):
                    X_labeled = X_train[train_idx]
                    y_labeled = y_train[train_idx]
                    X_unlabeled = np.delete(X_train, train_idx, axis=0)
                
                history = {
                    'acc': [],
                    'pseudo_counts': [],
                    'test_acc': []
                }
                
                # Vòng lặp Pseudo Label
                for iter in range(max_iter):
                    # Huấn luyện model
                    model = Sequential([
                        Dense(n_units, activation='relu', input_shape=(784,)),
                        Dropout(0.2),
                        Dense(10, activation='softmax')
                    ])
                    model.compile(Adam(0.001), 'sparse_categorical_crossentropy', metrics=['accuracy'])
                    
                    model.fit(X_labeled, y_labeled.astype(int), 
                             epochs=10, 
                             batch_size=128, 
                             verbose=0,
                             validation_split=0.2)
                    
                    # Dự đoán trên tập unlabeled
                    probas = model.predict(X_unlabeled, verbose=0)
                    pseudo_labels = np.argmax(probas, axis=1)
                    max_probs = np.max(probas, axis=1)
                    
                    # Chọn mẫu đạt ngưỡng
                    mask = max_probs >= threshold
                    X_selected = X_unlabeled[mask]
                    y_selected = pseudo_labels[mask]
                    
                    # Cập nhật dữ liệu
                    X_labeled = np.vstack([X_labeled, X_selected])
                    y_labeled = np.concatenate([y_labeled, y_selected])
                    X_unlabeled = X_unlabeled[~mask]
                    
                    # Đánh giá
                    test_loss, test_acc = model.evaluate(X_test, y_test.astype(int), verbose=0)
                    
                    # Lưu lịch sử
                    history['acc'].append(test_acc)
                    history['pseudo_counts'].append(len(X_selected))
                    history['test_acc'].append(test_acc)
                    
                    # Log metrics
                    mlflow.log_metrics({
                        f"iteration_{iter}_test_acc": test_acc,
                        f"iteration_{iter}_pseudo_added": len(X_selected)
                    }, step=iter)
                    
                    # Kiểm tra dừng
                    if len(X_unlabeled) == 0:
                        break
                
                # Visualization
                fig1 = px.line(
                    
                    x=list(range(1, len(history['acc'])+1),
                    y=history['acc'],
                    labels={'x': 'Vòng lặp', 'y': 'Độ chính xác'},
                    title='Độ chính xác qua các vòng lặp'
                )
                )
                st.plotly_chart(fig1)
                
                fig2 = px.bar(
                    x=list(range(1, len(history['pseudo_counts'])+1),
                    y=history['pseudo_counts'],
                    labels={'x': 'Vòng lặp', 'y': 'Số mẫu thêm vào'},
                    title='Số lượng pseudo labels thêm mỗi vòng'
                    )
                )
                st.plotly_chart(fig2)
                
# Tab Demo
def demo_tab():
    st.title("🎨 Demo Trực quan")
    
    # Load sample data
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
    sample_idx = np.random.choice(len(X), 5, replace=False)
    
    cols = st.columns(5)
    for i, col in enumerate(cols):
        with col:
            plt.imshow(X[sample_idx[i]].reshape(28, 28), cmap='gray')
            plt.axis('off')
            st.pyplot(plt)
            st.write(f"Nhãn thật: {y[sample_idx[i]]}")

# Main App
def main():
    st.set_page_config(page_title="Pseudo Labelling MNIST", page_icon="🔖")
    
    tab1, tab2, tab3 = st.tabs(["📚 Lý thuyết", "🔬 Thí nghiệm", "🎨 Demo"])
    
    with tab1:
        theory_tab()
    
    with tab2:
        experiment_tab()
    
    with tab3:
        demo_tab()

if __name__ == "__main__":
    main()