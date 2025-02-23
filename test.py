import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import joblib



# Khởi tạo MLflow
# mlflow.set_tracking_uri("sqlite:///mlruns.db")  # Lưu trữ local
# mlflow.set_experiment("MNIST Classification")

# Load dữ liệu MNIST
def ly_thuyet_Decision_tree():
    st.header("📖 Lý thuyết về Decision Tree")

    # 1️⃣ Giới thiệu về Decision Tree
    st.subheader("1️⃣ Giới thiệu về Decision Tree")
    st.write("""
    - **Decision Tree** hoạt động bằng cách chia nhỏ dữ liệu theo điều kiện để phân loại chính xác.
    - Mỗi nhánh trong cây là một câu hỏi "Có/Không" dựa trên đặc trưng dữ liệu.
    - Mô hình này dễ hiểu và trực quan nhưng có thể bị **overfitting** nếu không giới hạn độ sâu.
    """)

    # Hiển thị ảnh minh họa Decision Tree
    st.image("buoi4/img1.png", caption="Ví dụ về cách Decision Tree phân chia dữ liệu", use_container_width =True)

    st.write("""
    ### 🔍 Cách Decision Tree hoạt động với MNIST:
    - Mỗi ảnh trong MNIST có kích thước **28×28 pixels**, mỗi pixel có thể xem là một **đặc trưng (feature)**.
    - Mô hình sẽ quyết định phân tách dữ liệu bằng cách **chọn những pixels quan trọng nhất** để tạo nhánh.
    - Ví dụ, để phân biệt chữ số **0** và **1**, Decision Tree có thể kiểm tra:
        - Pixel ở giữa có sáng không?
        - Pixel dọc hai bên có sáng không?
    - Dựa trên câu trả lời, mô hình sẽ tiếp tục chia nhỏ tập dữ liệu.
    """)

    # 2️⃣ Công thức toán họ
    st.subheader("2️⃣ Các bước tính toán trong Decision Tree")

    st.markdown(r"""
    ### 📌 **Công thức chính**
    - **Entropy (Độ hỗn loạn của dữ liệu)**:
    $$
    H(S) = - \sum_{i=1}^{c} p_i \log_2 p_i
    $$
    → **Đo lường mức độ hỗn loạn của tập dữ liệu**. Nếu dữ liệu hoàn toàn đồng nhất, Entropy = 0. Nếu dữ liệu được phân bố đều giữa các lớp, Entropy đạt giá trị lớn nhất.

    **Trong đó:**  
    - \( c \) : số lượng lớp trong tập dữ liệu.  
    - \( $$p_i$$ \) : xác suất xuất hiện của lớp \( i \), được tính bằng tỷ lệ số mẫu của lớp \( i \) trên tổng số mẫu.

    - **Information Gain (Lợi ích thông tin sau khi chia tách)**:
    $$
    IG = H(S) - \sum_{j=1}^{k} \frac{|S_j|}{|S|} H(S_j)
    $$
    → **Đo lường mức độ giảm Entropy khi chia tập dữ liệu** theo một thuộc tính nào đó.  
    - Nếu **IG cao**, nghĩa là thuộc tính đó giúp phân loại tốt hơn.  
    - Nếu **IG thấp**, nghĩa là thuộc tính đó không có nhiều ý nghĩa để phân tách dữ liệu.

    **Trong đó:**  
    - \( S \) : tập dữ liệu ban đầu.  
    - \( $$S_j$$ \) : tập con sau khi chia theo thuộc tính đang xét.  
    - \( $$|S_j| / |S|$$ \) : tỷ lệ số lượng mẫu trong tập con \( $$S_j$$ \) so với tổng số mẫu.  
    - \( H(S) \) : Entropy của tập dữ liệu ban đầu.  
    - \( $$H(S_j)$$ \) : Entropy của tập con \( $$S_j$$ \).

    💡 **Cách áp dụng**:.
    
    1️⃣ **Tính Entropy \( H(S) \) của tập dữ liệu ban đầu**.  
    2️⃣ **Tính Entropy \( $$H(S_j)$$ \) của từng tập con khi chia theo từng thuộc tính**.  
    3️⃣ **Tính Information Gain cho mỗi thuộc tính**.  
    4️⃣ **Chọn thuộc tính có Information Gain cao nhất để chia nhánh**.  
    5️⃣ **Lặp lại quy trình trên cho đến khi dữ liệu được phân loại hoàn toàn**.  
    """)
    
    
    
def ly_thuyet_SVM():
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
    st.image("buoi4/img2.png", caption="SVM tìm siêu phẳng tối ưu để phân tách dữ liệu", use_container_width =True)

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
    $$\min_{w, b} \frac{1}{2} ||w||^2$$  
    → Mô hình cố gắng tìm **siêu phẳng phân cách** sao cho **vector trọng số \( w \) có độ lớn nhỏ nhất**, giúp tăng độ tổng quát.  

    **Trong đó:**  
    - \( w \) : vector trọng số xác định hướng của siêu phẳng.  
    - \( b \) : bias (độ dịch của siêu phẳng).  

    - **Ràng buộc**:  
    $$y_i (w \cdot x_i + b) \geq 1, \forall i$$  
    → Mọi điểm dữ liệu **phải nằm đúng phía** của siêu phẳng, đảm bảo phân loại chính xác.  

    **Trong đó:**  
    - \( $$xi$$ \) : điểm dữ liệu đầu vào.  
    - \( $$yi$$ \) : nhãn của điểm dữ liệu (\(+1\) hoặc \(-1\)).  

    - **Khoảng cách từ một điểm đến siêu phẳng**:  
    $$d = \frac{|w \cdot x + b|}{||w||}$$  
    → Đo **khoảng cách vuông góc** từ một điểm đến siêu phẳng, khoảng cách càng lớn thì mô hình càng đáng tin cậy.  

    - **Hàm mất mát với soft margin (SVM không tuyến tính)**:  
    $$\min_{w, b} \frac{1}{2} ||w||^2 + C \sum_{i=1}^{n} \xi_i$$  
    → Nếu dữ liệu **không thể phân tách hoàn hảo**, cho phép một số điểm bị phân loại sai với **biến slack \( $$\xi_i$$ \)**.  

    **Trong đó:**  
    - $$C$$ : hệ số điều chỉnh giữa việc tối ưu hóa margin và chấp nhận lỗi.  
    - $$\xi_i$$ : biến slack cho phép một số điểm bị phân loại sai.  
    """)

    st.write("""
    💡 **Ý nghĩa của công thức:**
    - SVM tối ưu hóa khoảng cách giữa hai lớp dữ liệu (margin).
    - Nếu dữ liệu không tuyến tính, kernel trick giúp ánh xạ dữ liệu lên không gian cao hơn.
    - \( C \) là hệ số điều chỉnh giữa việc tối ưu margin và chấp nhận lỗi.
    """)


def data():
    X = np.load("buoi4/X.npy")
    y = np.load("buoi4/y.npy")
    # Lọc các ảnh với nhãn 0 và 1
    idx_label_0 = np.where(y == 0)[0][:10]  # Lấy 10 ảnh với nhãn 0
    idx_label_1 = np.where(y == 1)[0][:10]  # Lấy 10 ảnh với nhãn 1

    # Lấy 10 ảnh với nhãn 0 và 10 ảnh với nhãn 1
    X_label_0 = X[idx_label_0]
    X_label_1 = X[idx_label_1]

    # Kết hợp cả 20 ảnh vào một danh sách
    X_combined = np.vstack((X_label_0, X_label_1))

    # Vẽ ảnh
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))  # Tạo lưới 10x10
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.imshow(X_combined[i].reshape(28, 28), cmap="gray")  # Hiển thị ảnh, giả sử ảnh là 28x28
        ax.axis("off")  # Tắt trục
        label = 0 if i < 10 else 1  # Gán nhãn (0 cho 10 ảnh đầu tiên, 1 cho 10 ảnh tiếp theo)
        ax.set_title(f"Label: {label}", fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)




def split_data():
    
    st.title("📌 Chia dữ liệu Train/Test")

    # Đọc dữ liệu
    X = np.load("buoi4/X.npy")
    y = np.load("buoi4/y.npy")
    total_samples = X.shape[0]

    # Thanh kéo chọn số lượng ảnh để train
    num_samples = st.slider("Chọn số lượng ảnh để train:", 1000, total_samples, 10000)

    # Thanh kéo chọn tỷ lệ Train/Test
    test_size = st.slider("Chọn tỷ lệ test:", 0.1, 0.5, 0.2)

    if st.button("✅ Xác nhận & Lưu"):
        # Lấy số lượng ảnh mong muốn
        X_selected, y_selected = X[:num_samples], y[:num_samples]

        # Chia train/test theo tỷ lệ đã chọn
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y_selected, test_size=test_size, random_state=42)

        # Lưu vào session_state để sử dụng sau
        st.session_state["X_train"] = X_train
        st.session_state["y_train"] = y_train
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test

        st.success(f"🔹 Dữ liệu đã được chia: Train ({len(X_train)}), Test ({len(X_test)})")

    # Kiểm tra nếu đã lưu dữ liệu vào session_state
    if "X_train" in st.session_state:
        st.write("📌 Dữ liệu train/test đã sẵn sàng để sử dụng!")
        
    
    
    
    
    
    
def train():
    # 📥 **Tải dữ liệu MNIST**
    if "X_train" in st.session_state:
        X_train = st.session_state["X_train"]
        # st.write(X_train.dtype)
        y_train = st.session_state["y_train"]
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]
    else:
        st.error("⚠️ Chưa có dữ liệu! Hãy chia dữ liệu trước.")
        return


    # 🌟 Chuẩn hóa dữ liệu
    
    X_train = X_train.reshape(-1, 28 * 28) / 255.0
    
    X_test = X_test.reshape(-1, 28 * 28) / 255.0


    st.header("⚙️ Chọn mô hình & Huấn luyện")

    # 📌 **Chọn mô hình**
    model_choice = st.selectbox("Chọn mô hình:", ["Decision Tree", "SVM"])

    if model_choice == "Decision Tree":
        st.markdown("""
        - **🌳 Decision Tree (Cây quyết định)** giúp chia dữ liệu thành các nhóm bằng cách đặt câu hỏi nhị phân dựa trên đặc trưng.
        - **Tham số cần chọn:**  
            - **max_depth**: Giới hạn độ sâu tối đa của cây.  
                - **Giá trị nhỏ**: Tránh overfitting nhưng có thể underfitting.  
                - **Giá trị lớn**: Cây có thể học tốt hơn nhưng dễ bị overfitting.  
        """)
        
        max_depth = st.slider("max_depth", 1, 20, 5)
        model = DecisionTreeClassifier(max_depth=max_depth)

    elif model_choice == "SVM":
        st.markdown("""
        - **🛠️ SVM (Support Vector Machine)** là mô hình tìm siêu phẳng tốt nhất để phân tách dữ liệu.
        - **Tham số cần chọn:**  
            - **C (Regularization)**: Hệ số điều chỉnh độ phạt lỗi.  
                - **C nhỏ**: Mô hình có thể bỏ qua một số lỗi nhưng tổng thể ổn định hơn.  
                - **C lớn**: Mô hình cố gắng phân loại chính xác từng điểm nhưng dễ bị overfitting.  
            - **Kernel**: Hàm ánh xạ dữ liệu lên không gian đặc trưng cao hơn.  
                - `"linear"` → Mô hình dùng siêu phẳng tuyến tính để phân lớp.  
                - `"rbf"` → Kernel Gaussian giúp phân tách dữ liệu phi tuyến tính tốt hơn.  
                - `"poly"` → Sử dụng đa thức bậc cao để phân lớp.  
                - `"sigmoid"` → Biến đổi giống như mạng nơ-ron nhân tạo.  
        """)
        C = st.slider("C (Regularization)", 0.1, 10.0, 1.0)
        kernel = st.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
        model = SVC(C=C, kernel=kernel)

    if st.button("Huấn luyện mô hình"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.success(f"✅ Độ chính xác: {acc:.4f}")

        # Lưu mô hình vào session_state dưới dạng danh sách nếu chưa có
        if "models" not in st.session_state:
            st.session_state["models"] = []

        # Tạo tên mô hình dựa trên lựa chọn mô hình và kernel
        model_name = model_choice.lower().replace(" ", "_")
        if model_choice == "SVM":
            model_name += f"_{kernel}"

        # Kiểm tra nếu tên mô hình đã tồn tại trong session_state
        existing_model = next((item for item in st.session_state["models"] if item["name"] == model_name), None)
        
        if existing_model:
            # Tạo tên mới với số đếm phía sau
            count = 1
            new_model_name = f"{model_name}_{count}"
            
            # Kiểm tra tên mới chưa tồn tại
            while any(item["name"] == new_model_name for item in st.session_state["models"]):
                count += 1
                new_model_name = f"{model_name}_{count}"
            
            # Sử dụng tên mới đã tạo
            model_name = new_model_name
            st.warning(f"⚠️ Mô hình được lưu với tên là: {model_name}")

        # Lưu mô hình vào danh sách với tên mô hình cụ thể
        st.session_state["models"].append({"name": model_name, "model": model})
        st.write(f"🔹 Mô hình đã được lưu với tên: {model_name}")
        st.write(f"Tổng số mô hình hiện tại: {len(st.session_state['models'])}")

        # In tên các mô hình đã lưu
        st.write("📋 Danh sách các mô hình đã lưu:")
        model_names = [model["name"] for model in st.session_state["models"]]
        st.write(", ".join(model_names))  # Hiển thị tên các mô hình trong một dòng
        

      

  
        
        

def load_model(path):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        st.error(f"⚠️ Không tìm thấy mô hình tại `{path}`")
        st.stop()

# ✅ Xử lý ảnh từ canvas (chuẩn 28x28 cho MNIST)
def preprocess_canvas_image(canvas_result):
    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data[:, :, 0].astype(np.uint8))
        img = img.resize((28, 28)).convert("L")  # Resize và chuyển thành grayscale
        img = np.array(img, dtype=np.float32) / 255.0  # Chuẩn hóa về [0, 1]
        return img.reshape(1, -1)  # Chuyển thành vector 1D
    return None

# ✅ Chạy dự đoán
def du_doan():
    st.header("✍️ Vẽ số để dự đoán")
    st.header("✍️ Vẽ số để dự đoán")
    st.header("✍️ Vẽ số để dự đoán")
    st.header("✍️ Vẽ số để dự đoán")
    # 🔹 Danh sách mô hình có sẵn
    models = {
        "SVM Linear": "buoi4/svm_mnist_linear.joblib",
        "SVM Poly": "buoi4/svm_mnist_poly.joblib",
        "SVM Sigmoid": "buoi4/svm_mnist_sigmoid.joblib",
        "SVM RBF": "buoi4/svm_mnist_rbf.joblib",
    }
    
    # Lấy tên mô hình từ session_state
    model_names = [model["name"] for model in st.session_state.get("models", [])]
    
    # 📌 Chọn mô hình
    model_option = st.selectbox("🔍 Chọn mô hình:", list(models.keys()) + model_names)

    # Nếu chọn mô hình đã được huấn luyện và lưu trong session_state
    if model_option in model_names:
        model = next(model for model in st.session_state["models"] if model["name"] == model_option)["model"]
    else:
        # Nếu chọn mô hình có sẵn (các mô hình đã được huấn luyện và lưu trữ dưới dạng file)
        model = load_model(models[model_option])
        st.success(f"✅ Đã tải mô hình: {model_option}")

    # ✍️ Vẽ số
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
        img = preprocess_canvas_image(canvas_result)

        if img is not None:
            # Hiển thị ảnh sau xử lý
            st.image(Image.fromarray((img.reshape(28, 28) * 255).astype(np.uint8)), caption="Ảnh sau xử lý", width=100)

            # Dự đoán
            prediction = model.predict(img)
            st.subheader(f"🔢 Dự đoán: {prediction[0]}")
        else:
            st.error("⚠️ Hãy vẽ một số trước khi bấm Dự đoán!")
            
            
            
            
            
            
            
def Classification2():
  

    st.title("🖊️ MNIST Classification App")

    ### **Phần 1: Hiển thị dữ liệu MNIST**
    
    ### **Phần 2: Trình bày lý thuyết về Decision Tree & SVM*
    
    # 1️⃣ Phần giới thiệu
    
    # === Sidebar để chọn trang ===
    # === Tạo Tabs ===
    tab1, tab2, tab3, tab4 = st.tabs(["📘 Lý thuyết Decision Tree", "📘 Lý thuyết SVM", "⚙️ Huấn luyện", "🔢 Dự đoán"])

    with tab1:
        ly_thuyet_Decision_tree()

    with tab2:
        ly_thuyet_SVM()
        data()
    with tab3:
        split_data()
        train()
        
    with tab4:
        du_doan()
        





            
if __name__ == "__main__":
    Classification2()