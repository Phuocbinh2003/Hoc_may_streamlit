import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import joblib  # Lưu mô hình

def show_explanations():
    st.markdown("""
    ## 📚 Giải thích Quy trình Tiền xử lý
    1. **Tải dữ liệu**: Nhập tập tin ảnh (.npy) và nhãn tương ứng  
    2. **Kiểm tra kích thước**: Đảm bảo số lượng ảnh và nhãn khớp nhau  
    3. **Làm phẳng ảnh**: Chuyển ảnh 2D (28x28) thành vector 1D (784 pixel)  
    4. **Mã hóa nhãn**: Chuyển đổi nhãn chữ cái thành số nguyên  
    5. **Chuẩn hóa dữ liệu**: Đưa giá trị pixel về khoảng [0,1]  
    """)

def display_sample_images(X, y, n_rows=3, n_cols=5):
    st.subheader("🖼️ Gallery Ảnh Mẫu")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 8))
    for i in range(n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        idx = np.random.randint(0, len(X))
        axes[row, col].imshow(X[idx].reshape(28, 28), cmap='gray')
        axes[row, col].set_title(f"Label: {y[idx]}", fontsize=8)
        axes[row, col].axis('off')
    plt.tight_layout()
    st.pyplot(fig)

def analyze_pixel_distribution(data):
    st.subheader("📈 Phân phối Giá trị Pixel")
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.hist(data.flatten(), bins=50, color='blue', alpha=0.7)
    plt.title('Phân phối gốc')
    plt.xlabel('Giá trị pixel')
    plt.ylabel('Tần suất')

    plt.subplot(1, 2, 2)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    plt.hist(scaled_data.flatten(), bins=50, color='green', alpha=0.7)
    plt.title('Sau chuẩn hóa')
    plt.xlabel('Giá trị pixel (0-1)')

    plt.tight_layout()
    st.pyplot(plt)

def main():
    st.title("🔠 Tiền Xử lý, Huấn luyện & Dự đoán ảnh chữ cái")

    show_explanations()

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
                st.error("❌ Số lượng ảnh và nhãn không khớp!")
                return

            # Tiền xử lý
            X_flat = X.reshape(X.shape[0], -1)
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X_flat)

            df = pd.DataFrame(X_scaled, columns=[f"pixel_{i}" for i in range(X_scaled.shape[1])])
            df['label'] = y_encoded

            # Tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🖼️ Ảnh & Phân tích", "📊 Dữ liệu", "🔤 Nhãn", "🤖 Huấn luyện", "🎯 Dự đoán"
            ])

            with tab1:
                st.subheader("Thông tin Dataset")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tổng số mẫu", len(X))
                with col2:
                    st.metric("Kích thước ảnh", f"{X.shape[1:] if X.ndim == 3 else X.shape[1]}")
                with col3:
                    st.metric("Số lớp", len(np.unique(y)))

                display_sample_images(X, y)
                analyze_pixel_distribution(X_flat)

            with tab2:
                st.dataframe(df.head(), use_container_width=True)

            with tab3:
                label_map = pd.DataFrame({
                    "Ký tự": le.classes_,
                    "Mã số": le.transform(le.classes_)
                })
                st.dataframe(label_map, hide_index=True)

            with tab4:
                st.subheader("🤖 Huấn luyện mô hình")

                algo = st.selectbox("Chọn thuật toán", ["Logistic Regression", "KNN"])
                n_samples = st.slider("Số lượng mẫu để huấn luyện", 100, len(X), 1000, step=100)
                test_size = st.slider("Tỷ lệ test", 0.1, 0.5, 0.2, 0.05)

                # Trích mẫu
                X_sample = X_scaled[:n_samples]
                y_sample = y_encoded[:n_samples]

                X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=test_size, random_state=42)

                if algo == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000)
                else:
                    k = st.slider("Số lượng hàng xóm (k)", 1, 15, 3)
                    model = KNeighborsClassifier(n_neighbors=k)

                if st.button("🚀 Huấn luyện"):
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)

                    joblib.dump(model, "trained_model.pkl")
                    joblib.dump(le, "label_encoder.pkl")
                    st.success(f"🎯 Độ chính xác: {acc * 100:.2f}%")

                    fig, ax = plt.subplots(figsize=(8, 6))
                    cm = confusion_matrix(y_test, y_pred)
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
                    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
                    st.pyplot(fig)

            with tab5:
                st.subheader("🎯 Dự đoán từ ảnh")
                if not joblib.os.path.exists("trained_model.pkl"):
                    st.warning("⚠️ Vui lòng huấn luyện mô hình trước!")
                else:
                    model = joblib.load("trained_model.pkl")
                    le = joblib.load("label_encoder.pkl")

                    test_idx = st.slider("Chọn chỉ số ảnh test", 0, len(X_scaled) - 1, 0)
                    image = X_scaled[test_idx].reshape(1, -1)
                    true_label = y[test_idx]

                    pred_label = le.inverse_transform(model.predict(image))[0]

                    st.image(X[test_idx], width=150, caption="Ảnh cần dự đoán")
                    st.write(f"🔍 **Dự đoán:** `{pred_label}`")
                    st.write(f"✅ **Nhãn thật:** `{true_label}`")

            # Tải xuống
            st.download_button(
                label="📥 Tải xuống Dataset đã xử lý",
                data=df.to_csv(index=False).encode(),
                file_name="processed_alphabet.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"⚠️ Lỗi xử lý: {str(e)}")
    else:
        st.info("👉 Vui lòng tải lên cả file ảnh và file nhãn để bắt đầu")

if __name__ == "__main__":
    main()
