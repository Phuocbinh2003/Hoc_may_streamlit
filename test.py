import os
import subprocess
import time
import mlflow
import streamlit as st
from mlflow.tracking import MlflowClient
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate

def appptest():
    # Đặt đường dẫn MLflow Tracking

    # 🌟 Cấu hình DAGsHub MLflow Tracking URI
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/Phuocbinh2003/Hoc_may_python.mlflow"
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)

    # Đăng nhập bằng username và token DAGsHub
    os.environ["MLFLOW_TRACKING_USERNAME"] = "Phuocbinh2003"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "c1495823c8f9156923b06f15899e989db7e62052"

    # 🎯 Tạo hoặc kết nối với Experiment
    experiment_name = "Streamlit-MLflow-Demo"
    mlflow.set_experiment(experiment_name)

    # 🏆 Giao diện Streamlit
    st.title("🚀 Streamlit + MLflow (DAGsHub)")

    # Nhập giá trị learning rate
    learning_rate = st.slider("Chọn Learning Rate:", 0.001, 0.1, 0.01)

    # Nhập giá trị accuracy
    accuracy = st.slider("Chọn Accuracy:", 0.80, 1.00, 0.95)

    if st.button("Ghi log lên MLflow 🚀"):
        with mlflow.start_run():
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_metric("accuracy", accuracy)

        st.success("✅ Đã ghi log lên DAGsHub MLflow!")
        st.markdown(f"🔗 **Xem logs tại DAGsHub MLflow:** [Click here]({DAGSHUB_MLFLOW_URI}/experiments/)")



# Gọi hàm khi chạy ứng dụng Streamlit
if __name__ == "__main__":
    appptest()
