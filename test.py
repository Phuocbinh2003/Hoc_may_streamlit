import os
import subprocess
import time
import mlflow
import streamlit as st
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
def appptest():
    
    # Đặt sẵn API Key của OpenAI (hoặc bạn có thể lấy từ biến môi trường)
    os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

    # Khởi tạo MLflow với experiment ID
    experiment_id = "251068899510733485"
    mlflow.set_experiment(experiment_id=experiment_id)

    # Hàm khởi động MLflow UI trong nền
    def start_mlflow_ui():
        try:
            subprocess.Popen(["mlflow", "ui", "--port", "5000"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(3)  # Chờ vài giây để UI khởi động
        except Exception as e:
            st.error(f"Không thể khởi động MLflow UI: {e}")

    # Chạy MLflow UI khi mở ứng dụng Streamlit
    start_mlflow_ui()

    st.title("MLflow LangChain Tracking với Streamlit")

    # Hiển thị link truy cập MLflow UI
    st.markdown("### 🔗 [Truy cập MLflow UI](http://localhost:5000)")

    # Gọi LangChain để ghi log vào MLflow
    with mlflow.start_run():
        llm = OpenAI()
        prompt = PromptTemplate.from_template("Answer the following question: {question}")
        chain = prompt | llm

        # Câu hỏi demo
        question = st.text_input("Nhập câu hỏi:", "What is MLflow?")
        
        if st.button("Gửi câu hỏi"):
            response = chain.invoke(question)

            # Ghi log vào MLflow
            mlflow.log_param("prompt", "Answer the following question: {question}")
            mlflow.log_param("question", question)
            mlflow.log_param("model", "OpenAI GPT")
            mlflow.log_metric("response_length", len(response))
            mlflow.log_text(response, "response.txt")

            st.write("### Phản hồi từ mô hình:")
            st.write(response)
