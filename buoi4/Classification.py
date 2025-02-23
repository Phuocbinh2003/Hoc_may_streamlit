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


def Classification():
    digits = datasets.load_digits()
    X, y = digits.data, digits.target

    # Chia tập train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
    st.image("buoi4/img2.png", caption="SVM tìm siêu phẳng tối ưu để phân tách dữ liệu", use_column_width=True)

    st.write("""
    ### 🔍 **Cách hoạt động của SVM**
    - Dữ liệu được biểu diễn trong không gian nhiều chiều.
    - Mô hình tìm một siêu phẳng để phân tách dữ liệu sao cho khoảng cách từ siêu phẳng đến các điểm gần nhất (support vectors) là lớn nhất.
    - Nếu dữ liệu **không thể phân tách tuyến tính**, ta có thể:
        - **Dùng Kernel Trick** để ánh xạ dữ liệu sang không gian cao hơn.
        - **Thêm soft margin** để chấp nhận một số điểm bị phân loại sai.
    """)




    st.subheader("📌 Công thức toán học")

    st.write("### 1️⃣ Hàm mục tiêu cần tối ưu")
    st.latex(r"\min_{w, b} \frac{1}{2} ||w||^2")
    st.write("""
    💡 **Ý nghĩa**:  
    - Đây là **hàm mất mát** của SVM, mục tiêu là tìm **vector trọng số** \( w \) có độ dài nhỏ nhất nhưng vẫn đảm bảo phân tách hai lớp dữ liệu một cách tốt nhất.  
    - Giá trị **\( \frac{1}{2} ||w||^2 \)** thể hiện mức độ phức tạp của siêu phẳng, cần tối thiểu hóa nó để có mô hình đơn giản và hiệu quả.  
    """)

    st.write("### 2️⃣ Điều kiện ràng buộc")
    st.latex(r"y_i (w \cdot x_i + b) \geq 1, \forall i")
    st.write("""
    💡 **Ý nghĩa**:  
    - Điều kiện này đảm bảo **tất cả điểm dữ liệu** nằm đúng bên siêu phẳng phân cách.  
    - Với mỗi điểm dữ liệu \( x_i \):  
    - Nếu \( y_i = 1 \), tức là thuộc nhóm **dương**, thì \( w \cdot x_i + b \) phải lớn hơn 1.  
    - Nếu \( y_i = -1 \), tức là thuộc nhóm **âm**, thì \( w \cdot x_i + b \) phải nhỏ hơn -1.  
    - **Khoảng cách từ siêu phẳng đến điểm gần nhất được tối ưu**, giúp mô hình có độ tổng quát cao.  
    """)

    st.write("### 3️⃣ Khoảng cách từ một điểm đến siêu phẳng")
    st.latex(r"d = \frac{|w \cdot x + b|}{||w||}")
    st.write("""
    💡 **Ý nghĩa**:  
    - Công thức này tính **khoảng cách vuông góc từ một điểm đến siêu phẳng phân cách**.  
    - Một mô hình SVM tốt sẽ chọn siêu phẳng sao cho khoảng cách này **lớn nhất** với các điểm dữ liệu gần nhất (support vectors).  
    - Nếu khoảng cách này nhỏ, có nghĩa là mô hình có thể dễ bị lỗi khi gặp dữ liệu mới.  
    """)

    st.write("### 4️⃣ Hàm mất mát với soft margin (cho dữ liệu không hoàn toàn tách tuyến tính)")
    st.latex(r"\min_{w, b} \frac{1}{2} ||w||^2 + C \sum_{i=1}^{n} \xi_i")
    st.write("""
    💡 **Ý nghĩa**:  
    - Khi dữ liệu **không thể phân tách hoàn hảo**, ta cần cho phép một số điểm nằm sai bên siêu phẳng (sai số nhỏ).  
    - **\( \xi_i \)** là **biến slack**, cho phép một số điểm bị phân loại sai nhưng vẫn cố gắng tối ưu hóa khoảng cách.  
    - **\( C \)** là một tham số điều chỉnh:  
    - Nếu **\( C \) lớn**, mô hình cố gắng phân loại chính xác tất cả điểm nhưng có thể bị overfitting.  
    - Nếu **\( C \) nhỏ**, mô hình có thể bỏ qua một số lỗi nhỏ nhưng có khả năng tổng quát tốt hơn.  
    """)

    st.write("""
    ### 🔥 **Tóm tắt**
    - **SVM tìm một siêu phẳng tối ưu** để phân loại dữ liệu với khoảng cách lớn nhất đến các điểm gần nhất.
    - **Nếu dữ liệu không tuyến tính**, ta có thể dùng **kernel trick** để ánh xạ lên không gian cao hơn.
    - **Tham số \( C \)** giúp điều chỉnh giữa độ chính xác và khả năng tổng quát.
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