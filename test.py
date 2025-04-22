import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# ----------- Giải thích quy trình ----------------
def show_explanations():
    st.markdown("""
    ## 📚 Giải thích Quy trình Tiền xử lý.
    1. **Tải dữ liệu**: Nhập tập tin ảnh (.npy) và nhãn tương ứng  
    2. **Kiểm tra kích thước**: Đảm bảo số lượng ảnh và nhãn khớp nhau  
    3. **Làm phẳng ảnh**: Chuyển ảnh 2D (28x28) thành vector 1D  
    4. **Mã hóa nhãn**: Chuyển đổi nhãn chữ cái thành số nguyên  
    5. **Chuẩn hóa dữ liệu**: Đưa giá trị pixel về khoảng [0,1]  
    6. **Huấn luyện mô hình**: Logistic Regression hoặc KNN  
    7. **Dự đoán & Đánh giá**: Độ chính xác + Ma trận nhầm lẫn
    """)

# ----------- Hiển thị ảnh mẫu ----------------
def display_sample_images(X, y, n_rows=3, n_cols=5):
    st.subheader("🖼️ Gallery Ảnh Mẫu")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 8))
    for i in range(n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        idx = np.random.randint(0, len(X))
        try:
            img = X[idx].reshape(28, 28)
            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].set_title(f"Label: {y[idx]}", fontsize=8)
            axes[row, col].axis('off')
        except Exception as e:
            st.warning(f"Lỗi khi hiển thị ảnh: {e}")
    st.pyplot(fig)
    plt.close()

# ----------- Phân phối pixel ----------------
def analyze_pixel_distribution(X_flat):
    st.subheader("📈 Phân phối Giá trị Pixel")
    plt.figure(figsize=(10, 4))

    # Trước chuẩn hóa
    plt.subplot(1, 2, 1)
    plt.hist(X_flat.flatten(), bins=50, color='blue', alpha=0.7)
    plt.title('Phân phối gốc')
    plt.xlabel('Pixel')
    plt.ylabel('Tần suất')

    # Sau chuẩn hóa
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(X_flat)

    plt.subplot(1, 2, 2)
    plt.hist(scaled.flatten(), bins=50, color='green', alpha=0.7)
    plt.title('Sau chuẩn hóa')
    plt.xlabel('Pixel (0-1)')

    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

# ----------- Main App ----------------
def main():
    st.set_page_config(page_title="Alphabet Image Preprocessing", layout="wide")
    st.title("🔠 Tiền Xử lý & Huấn luyện Ảnh Chữ cái")
    show_explanations()

    # Upload file
    with st.expander("📤 Tải lên Dữ liệu", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            X_file = st.file_uploader("Chọn file ảnh (.npy)", type="npy")
        with col2:
            y_file = st.file_uploader("Chọn file nhãn (.npy)", type="npy")

    if X_file and y_file:
        try:
            X = np.load(X_file)
            y = np.load(y_file).astype(str)

            if len(X) != len(y):
                st.error("❌ Số lượng ảnh và nhãn không khớp.")
                return
            if X.shape[1:] != (28, 28):
                st.error("⚠️ Ảnh cần có kích thước (28x28).")
                return

            st.subheader("📦 Thông tin Dataset")
            st.write(f"Số mẫu: {len(X)}")
            st.write(f"Kích thước ảnh: {X.shape[1:]}")

            display_sample_images(X, y)

            X_flat = X.reshape(X.shape[0], -1)
            analyze_pixel_distribution(X_flat)

            # Label encoding & normalization
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X_flat)

            df = pd.DataFrame(X_scaled)
            df['label'] = y_encoded

            # Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Dữ liệu", "🔤 Nhãn", "📈 Phân phối", "🧠 Huấn luyện"])

            with tab1:
                st.dataframe(df.head(), use_container_width=True)

            with tab2:
                st.dataframe(pd.DataFrame({"Ký tự": le.classes_, "Mã số": le.transform(le.classes_)}))

            with tab3:
                st.write("Phân phối lớp:")
                class_dist = pd.Series(y).value_counts().reset_index()
                class_dist.columns = ['Ký tự', 'Số lượng']
                st.bar_chart(class_dist.set_index('Ký tự'))

            with tab4:
                st.subheader("🔧 Cấu hình Huấn luyện")
                model_type = st.selectbox("Chọn mô hình", ["Logistic Regression", "KNN"])
                test_size = st.slider("Tỉ lệ Test", 0.1, 0.5, 0.2, step=0.05)

                if model_type == "KNN":
                    n_neighbors = st.slider("Số láng giềng (K)", 1, 15, 3)

                if st.button("🚀 Bắt đầu Huấn luyện"):
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y_encoded, test_size=test_size, random_state=42)

                    model = (LogisticRegression(max_iter=1000)
                             if model_type == "Logistic Regression"
                             else KNeighborsClassifier(n_neighbors=n_neighbors))

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    st.success(f"🎯 Độ chính xác: {acc*100:.2f}%")

                    # Confusion matrix
                    cm = confusion_matrix(y_test, y_pred)
                    labels_present = le.inverse_transform(np.unique(y_test))
                    fig, ax = plt.subplots(figsize=(8, 6))
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_present)
                    disp.plot(ax=ax, cmap='Blues')
                    st.pyplot(fig)
                    plt.close()

                    # Dự đoán ảnh mới
                    st.subheader("🔍 Dự đoán Ảnh Mới")
                    uploaded = st.file_uploader("Tải ảnh .npy để dự đoán", type="npy")
                    if uploaded:
                        new_img = np.load(uploaded)
                        if new_img.shape == (28, 28):
                            new_flat = new_img.reshape(1, -1)
                            new_scaled = scaler.transform(new_flat)
                            pred = model.predict(new_scaled)
                            st.image(new_img, caption="Ảnh nhập", width=150)
                            st.success(f"✅ Dự đoán: {le.inverse_transform(pred)[0]}")
                        else:
                            st.error("⚠️ Ảnh phải có kích thước 28x28.")

            # Download
            st.download_button("📥 Tải dataset đã xử lý", df.to_csv(index=False).encode(), "processed.csv", "text/csv")

        except Exception as e:
            st.error(f"⚠️ Lỗi xử lý: {str(e)}")

    else:
        st.info("📌 Vui lòng tải lên cả ảnh và nhãn để bắt đầu.")

if __name__ == "__main__":
    main()
