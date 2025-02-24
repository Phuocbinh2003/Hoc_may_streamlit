import streamlit as st
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def appptest():
    
    # Kết nối với MLflow local
    mlflow.set_tracking_uri("http://localhost:5000")

    # Dữ liệu mẫu
    X, y = make_classification(n_samples=1000, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model
    model = LogisticRegression()

    # Giao diện Streamlit
    st.title("MLflow + Streamlit Demo")

    if st.button("Train Model & Log to MLflow"):
        with mlflow.start_run():  # Bắt đầu tracking với MLflow
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            # Log các thông số
            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("num_features", X.shape[1])
            mlflow.log_metric("accuracy", acc)
            
            # Lưu model vào MLflow
            mlflow.sklearn.log_model(model, "model")

            st.success(f"Model trained with accuracy: {acc:.4f}")
            st.write("Check MLflow at: [localhost:5000](http://localhost:5000)")

