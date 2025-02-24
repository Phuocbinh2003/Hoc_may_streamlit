import streamlit as st
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def appptest():
    


    # Kết nối với MLflow Tracking Server
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Tạo dữ liệu giả lập
    X, y = make_classification(n_samples=500, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.title("MLflow + Streamlit Demo")

    # Người dùng chọn tham số mô hình
    lr = st.slider("Learning Rate", 0.01, 1.0, 0.1)
    max_iter = st.slider("Max Iterations", 100, 1000, 200)

    # Bắt đầu log với MLflow
    with mlflow.start_run():
        # Log tham số
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("max_iterations", max_iter)

        # Train mô hình
        model = LogisticRegression(C=lr, max_iter=max_iter)
        model.fit(X_train, y_train)

        # Dự đoán và tính accuracy
        acc = model.score(X_test, y_test)
        st.write(f"Test Accuracy: {acc:.4f}")

        # Log metric
        mlflow.log_metric("accuracy", acc)

        # Log mô hình
        mlflow.sklearn.log_model(model, "logistic_regression_model")

    st.success("Mô hình đã được log lên MLflow!")

