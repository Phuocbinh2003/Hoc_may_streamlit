import streamlit as st
import mlflow
import mlflow.sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Khởi tạo MLflow
# mlflow.set_tracking_uri("sqlite:///mlruns.db")  # Lưu trữ local
# mlflow.set_experiment("MNIST Classification")

# Load dữ liệu MNIST
digits = datasets.load_digits()
X, y = digits.data, digits.target

# Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def Classification():
    # Giao diện Streamlit
    st.title("MNIST Digit Classification")

    # Chọn mô hình
    model_choice = st.selectbox("Chọn mô hình:", ["Decision Tree", "SVM"])

    # Hiển thị thanh điều chỉnh tham số theo từng mô hình
    if model_choice == "Decision Tree":
        max_depth = st.slider("max_depth", 1, 20, 5)
        min_samples_split = st.slider("min_samples_split", 2, 10, 2)
    else:
        C = st.slider("C (Regularization)", 0.1, 10.0, 1.0)
        kernel = st.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])

    if st.button("Huấn luyện"):
        with mlflow.start_run():
            if model_choice == "Decision Tree":
                model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
                mlflow.log_param("max_depth", max_depth)
                mlflow.log_param("min_samples_split", min_samples_split)
            else:
                model = SVC(C=C, kernel=kernel)
                mlflow.log_param("C", C)
                mlflow.log_param("kernel", kernel)

            # Huấn luyện mô hình
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            # Lưu kết quả vào MLflow
            mlflow.log_param("model", model_choice)
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(model, model_choice)

            # Hiển thị kết quả
            st.write(f"Độ chính xác của {model_choice}: {acc:.4f}")

            # Dự đoán một số ảnh
            st.subheader("Kết quả dự đoán trên tập test")
            fig, axes = plt.subplots(1, 5, figsize=(10, 3))
            for i, ax in enumerate(axes):
                ax.imshow(X_test[i].reshape(8, 8), cmap="gray")
                ax.set_title(f"Pred: {y_pred[i]}")
                ax.axis("off")
            st.pyplot(fig)
            
if __name__ == "__main__":
    Classification()