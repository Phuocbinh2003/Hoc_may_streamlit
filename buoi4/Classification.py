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
    # 1️⃣ Phần giới thiệu
    st.header("📖 Lý thuyết về Decision Tree")

    # 1️⃣ Giới thiệu về Decision Tree
    st.subheader("1️⃣ Giới thiệu về Decision Tree")
    st.write("""
    - **Decision Tree** hoạt động bằng cách chia nhỏ dữ liệu theo điều kiện để phân loại chính xác.
    - Mỗi nhánh trong cây là một câu hỏi "Có/Không" dựa trên đặc trưng dữ liệu.
    - Mô hình này dễ hiểu và trực quan nhưng có thể bị **overfitting** nếu không giới hạn độ sâu.
    """)

    # Hiển thị ảnh minh họa Decision Tree
    st.image("buoi4/img1.png", caption="Ví dụ về cách Decision Tree phân chia dữ liệu", use_column_width=True)

    st.write("""
    ### 🔍 Cách Decision Tree hoạt động với MNIST:
    - Mỗi ảnh trong MNIST có kích thước **28×28 pixels**, mỗi pixel có thể xem là một **đặc trưng (feature)**.
    - Mô hình sẽ quyết định phân tách dữ liệu bằng cách **chọn những pixels quan trọng nhất** để tạo nhánh.
    - Ví dụ, để phân biệt chữ số **0** và **1**, Decision Tree có thể kiểm tra:
        - Pixel ở giữa có sáng không?
        - Pixel dọc hai bên có sáng không?
    - Dựa trên câu trả lời, mô hình sẽ tiếp tục chia nhỏ tập dữ liệu.
    """)

    # 2️⃣ Công thức toán học
    st.subheader("2️⃣ Các bước tính toán trong Decision Tree")

    st.markdown(r"""
    ### 📌 **Công thức chính**
    - **Entropy (Độ hỗn loạn của dữ liệu)**:
    \[
    H(S) = - \sum_{i=1}^{c} p_i \log_2 p_i
    \]
    Trong đó:
    - \( c \) là số lượng lớp.
    - \( p_i \) là xác suất xuất hiện của lớp \( i \).

    - **Information Gain (Lợi ích thông tin sau khi chia tách)**:
    \[
    IG = H(S) - \sum_{j=1}^{k} \frac{|S_j|}{|S|} H(S_j)
    \]

    - **Gini Impurity (Thước đo thay thế Entropy)**:
    \[
    Gini(S) = 1 - \sum_{i=1}^{c} p_i^2
    \]

    💡 **Sau khi tính toán Entropy, mô hình chọn đặc trưng tốt nhất làm gốc, rồi tính Information Gain của các đặc trưng còn lại để tìm nhánh tiếp theo.**
    """)




    st.subheader("2️⃣ Support Vector Machine (SVM)")

    st.write("""
    - **Support Vector Machine (SVM)** là một thuật toán học máy mạnh mẽ để phân loại dữ liệu.
    - **Mục tiêu chính**: Tìm một **siêu phẳng (hyperplane)** tối ưu để phân tách các lớp dữ liệu.
    - **Ứng dụng**: Nhận diện khuôn mặt, phát hiện thư rác, phân loại văn bản, v.v.
    - **Ưu điểm**:
        - Hiệu quả trên dữ liệu có độ nhiễu thấp.
        - Hỗ trợ dữ liệu không tuyến tính bằng **kernel trick**.
    - **Nhược điểm**:
        - Chậm trên tập dữ liệu lớn do tính toán phức tạp.
        - Nhạy cảm với lựa chọn tham số (C, Kernel).
    """)

    # Hiển thị hình ảnh minh họa SV
    #st.image("buoi4/svm_example.png", caption="SVM tìm siêu phẳng tối ưu để phân tách dữ liệu", use_column_width=True)

    st.write("""
    ### 🔍 **Cách hoạt động của SVM**
    - Dữ liệu được biểu diễn trong không gian nhiều chiều.
    - Mô hình tìm một siêu phẳng để phân tách dữ liệu sao cho khoảng cách từ siêu phẳng đến các điểm gần nhất (support vectors) là lớn nhất.
    - Nếu dữ liệu **không thể phân tách tuyến tính**, ta có thể:
        - **Dùng Kernel Trick** để ánh xạ dữ liệu sang không gian cao hơn.
        - **Thêm soft margin** để chấp nhận một số điểm bị phân loại sai.
    """)

    # 📌 2️⃣ Công thức toán học
    st.subheader("📌 Công thức toán học")

    st.markdown(r"""
    - **Hàm mục tiêu cần tối ưu**:
    \[
    \min_{w, b} \frac{1}{2} ||w||^2
    \]
    với ràng buộc:
    \[
    y_i (w \cdot x_i + b) \geq 1, \forall i
    \]
    Trong đó:
    - \( w \) là vector trọng số.
    - \( b \) là bias (độ dịch của siêu phẳng).
    - \( x_i \) là điểm dữ liệu.
    - \( y_i \) là nhãn của điểm dữ liệu (\(+1\) hoặc \(-1\)).

    - **Khoảng cách từ một điểm đến siêu phẳng**:
    \[
    d = \frac{|w \cdot x + b|}{||w||}
    \]
    
    - **Hàm mất mát với soft margin (SVM không tuyến tính)**:
    \[
    \min_{w, b} \frac{1}{2} ||w||^2 + C \sum_{i=1}^{n} \xi_i
    \]
    với \( \xi_i \) là biến slack cho phép phân loại sai một số điểm.
    """)

    st.write("""
    💡 **Ý nghĩa của công thức:**
    - SVM tối ưu hóa khoảng cách giữa hai lớp dữ liệu (margin).
    - Nếu dữ liệu không tuyến tính, kernel trick giúp ánh xạ dữ liệu lên không gian cao hơn.
    - \( C \) là hệ số điều chỉnh giữa việc tối ưu margin và chấp nhận lỗi.
    """)

    # 📌 3️⃣ Ví dụ tính toán khoảng cách đến siêu phẳng
    st.subheader("📌 Ví dụ tính toán")

    # Mô phỏng dữ liệu đơn giản
    w = np.array([2, -3])  # Trọng số w
    b = 5  # Bias
    x_sample = np.array([1, 2])  # Một điểm dữ liệu

    # Tính khoảng cách đến siêu phẳng
    distance = np.abs(np.dot(w, x_sample) + b) / np.linalg.norm(w)

    st.write(f"📌 **Khoảng cách từ điểm {x_sample} đến siêu phẳng**: {distance:.4f}")

    # 📌 4️⃣ Minh họa phân tách dữ liệu bằng SVM
    st.subheader("📌 Minh họa phân tách dữ liệu")

    # Tạo dữ liệu mô phỏng
    X, y = make_classification(n_samples=100, n_features=2, n_classes=2, n_clusters_per_class=1, n_redundant=0, random_state=42)
    svm_model = SVC(kernel="linear", C=1.0)
    svm_model.fit(X, y)

    # Vẽ siêu phẳng
    fig, ax = plt.subplots(figsize=(6, 4))

    # Vẽ dữ liệu
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k")

    # Vẽ siêu phẳng
    xx = np.linspace(X[:, 0].min(), X[:, 0].max(), 50)
    yy = - (svm_model.coef_[0][0] * xx + svm_model.intercept_[0]) / svm_model.coef_[0][1]
    ax.plot(xx, yy, "k--", label="Siêu phẳng")

    # Vẽ support vectors
    ax.scatter(svm_model.support_vectors_[:, 0], svm_model.support_vectors_[:, 1], facecolors="none", edgecolors="k", s=100, label="Support Vectors")

    ax.set_title("Minh họa SVM với dữ liệu đơn giản")
    ax.legend()
    st.pyplot(fig)

    st.write("""
    ### 🔥 **Tóm tắt**
    - SVM tìm siêu phẳng tối ưu để phân loại dữ liệu.
    - Nếu dữ liệu không tuyến tính, có thể dùng **kernel trick**.
    - Cần chọn tham số **C, kernel** phù hợp để tránh overfitting.

    🚀 **Bạn có muốn thử nghiệm với dữ liệu thực tế?**
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