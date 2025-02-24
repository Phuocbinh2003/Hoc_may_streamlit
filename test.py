import os
import subprocess
import time
import mlflow
import streamlit as st
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate

def appptest():
    
    # Đảm bảo API Key của OpenAI có sẵn trong môi trường
    os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

    # Tạo hoặc đặt Experiment Name (sẽ tự động tạo nếu chưa có)
    mlflow.set_experiment("LangChain_Tracking")

    # Hàm kiểm tra xem MLflow UI đã chạy chưa
    def is_mlflow_running():
        try:
            output = subprocess.check_output("lsof -i :5000", shell=True).decode()
            return "mlflow" in output
        except subprocess.CalledProcessError:
            return False

    # Hàm khởi động MLflow UI nếu chưa chạy
    def start_mlflow_ui():
        if not is_mlflow_running():
            subprocess.Popen(["mlflow", "ui", "--port", "5000"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(3)  # Chờ vài giây để UI khởi động

    # Chạy MLflow UI khi mở ứng dụng Streamlit
    start_mlflow_ui()

    # Giao diện Streamlit
    st.title("MLflow LangChain Tracking với Streamlit")

    # Hiển thị link truy cập MLflow UI
    st.markdown("### 🔗 [Truy cập MLflow UI](http://localhost:5000)")

    # Nhập câu hỏi
    question = st.text_input("Nhập câu hỏi:", "What is MLflow?")

    if st.button("Gửi câu hỏi"):
        with mlflow.start_run():
            # Gọi mô hình LangChain
            llm = OpenAI()
            prompt = PromptTemplate.from_template("Answer the following question: {question}")
            chain = prompt | llm
            response = chain.invoke(question)

            # Ghi log vào MLflow
            mlflow.log_param("prompt", "Answer the following question: {question}")
            mlflow.log_param("question", question)
            mlflow.log_param("model", "OpenAI GPT")
            mlflow.log_metric("response_length", len(response))
            mlflow.log_text(response, "response.txt")

            # Hiển thị phản hồi từ mô hình
            st.write("### Phản hồi từ mô hình:")
            st.write(response)
