import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps

# Khởi tạo MLflow
# mlflow.set_tracking_uri("sqlite:///mlruns.db")  # Lưu trữ local
# mlflow.set_experiment("MNIST Classification")

# Load dữ liệu MNIST
digits = datasets.load_digits()
X, y = digits.data, digits.target

# Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def Classification():
    st.title("🖊️ MNIST Classification App")

    ### **Phần 1: Hiển thị dữ liệu MNIST**
    st.header("📊 Một số hình ảnh trong tập MNIST")
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(X[i].reshape(8, 8), cmap="gray")
        ax.set_title(f"Số {y[i]}")
        ax.axis("off")
    st.pyplot(fig)

    ### **Phần 2: Trình bày lý thuyết về Decision Tree & SVM**
    st.header("📖 Lý thuyết về mô hình")
    st.subheader("1️⃣ Decision Tree")
    st.write("""
    - **Decision Tree** hoạt động bằng cách chia nhỏ dữ liệu theo các điều kiện để phân loại chính xác.  
    - Mô hình có thể dễ hiểu nhưng dễ bị overfitting nếu không giới hạn độ sâu.
    """)

    st.subheader("2️⃣ SVM (Support Vector Machine)")
    st.write("""
    - **SVM** tìm một siêu phẳng để phân tách dữ liệu một cách tối ưu.  
    - SVM hiệu quả trên dữ liệu phức tạp nhưng có thể chậm trên dữ liệu lớn.
    """)

    ### **Phần 3: Chọn mô hình & Train**
    st.header("⚙️ Chọn mô hình & Huấn luyện")

    model_choice = st.selectbox("Chọn mô hình:", ["Decision Tree", "SVM"])

    if model_choice == "Decision Tree":
        max_depth = st.slider("max_depth", 1, 20, 5)
        model = DecisionTreeClassifier(max_depth=max_depth)
    elif model_choice == "SVM":
        C = st.slider("C (Regularization)", 0.1, 10.0, 1.0)
        kernel = st.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
        model = SVC(C=C, kernel=kernel)

    if st.button("Huấn luyện mô hình"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.success(f"✅ Độ chính xác: {acc:.4f}")

        # Lưu kết quả vào MLflow
        with mlflow.start_run():
            mlflow.log_param("model", model_choice)
            if model_choice == "Decision Tree":
                mlflow.log_param("max_depth", max_depth)
            else:
                mlflow.log_param("C", C)
                mlflow.log_param("kernel", kernel)
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(model, model_choice)

    ### **Phần 4: Vẽ số & Dự đoán**
    st.header("✍️ Vẽ số để dự đoán")

    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=10,
        stroke_color="white",
        background_color="black",
        height=150,
        width=150,
        drawing_mode="freedraw",
        key="canvas"
    )

    if st.button("Dự đoán số"):
        if canvas_result.image_data is not None:
            img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype(np.uint8))
            img = img.resize((8, 8)).convert("L")
            img = ImageOps.invert(img)
            img = np.array(img).reshape(1, -1)

            # Dự đoán
            prediction = model.predict(img)
            st.subheader(f"🔢 Dự đoán: {prediction[0]}")
            
if __name__ == "__main__":
    Classification()