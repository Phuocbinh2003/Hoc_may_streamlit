import os
import mlflow
import streamlit as st
from mlflow.tracking import MlflowClient

def appptest():
    st.title("🚀 MLflow DAGsHub Tracking với Streamlit")

    # 🌟 Cấu hình DAGsHub MLflow Tracking URI
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/Phuocbinh2003/Hoc_may_python.mlflow"
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)

    # Đăng nhập bằng username và token DAGsHub
    os.environ["MLFLOW_TRACKING_USERNAME"] = "Phuocbinh2003"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "c1495823c8f9156923b06f15899e989db7e62052"

    # 📝 Kiểm tra danh sách các experiment có sẵn
    client = MlflowClient()
    experiments = client.search_experiments()

    # Hiển thị danh sách experiment
    st.subheader("📌 Danh sách Experiments:")
    for exp in experiments:
        st.write(f"- ID: {exp.experiment_id}, Name: {exp.name}")

    # Chọn hoặc tạo experiment
    experiment_name = "My_Experiment"
    experiment = mlflow.set_experiment(experiment_name)

    # Ghi log vào MLflow
    with mlflow.start_run():
        mlflow.log_param("model", "OpenAI GPT")
        mlflow.log_param("task", "Streamlit DAGsHub MLflow")
        mlflow.log_metric("accuracy", 0.95)

    st.success("✅ Đã log dữ liệu vào MLflow DAGsHub!")

    # Hiển thị link truy cập MLflow trên DAGsHub
    st.markdown(f"### 🔗 [Truy cập MLflow DAGsHub]({DAGSHUB_MLFLOW_URI})")

# Chạy app Streamlit
if __name__ == "__main__":
    appptest()
