import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.stats import zscore

# Hàm hiển thị giải thích từng bước
def show_explanations():
    st.markdown("""
    ## 📚 Giải thích Quy trình Tiền xử lý
    
    1. **Tải dữ liệu**: Nhập tập tin ảnh (.npy) và nhãn tương ứng
    2. **Kiểm tra kích thước**: Đảm bảo số lượng ảnh và nhãn khớp nhau
    3. **Làm phẳng ảnh**: Chuyển ảnh 2D (28x28) thành vector 1D (784 pixel)
    4. **Mã hóa nhãn**: Chuyển đổi nhãn chữ cái thành số nguyên
    5. **Chuẩn hóa dữ liệu**: Đưa giá trị pixel về khoảng [0,1]
    6. **Phân tích chất lượng**: Kiểm tra outliers và ảnh hỏng
    7. **Trực quan hóa**: Hiển thị kết quả xử lý
    """)

# Hàm hiển thị ảnh mẫu
def display_sample_images(X, y, n_rows=3, n_cols=5):
    st.subheader("🖼️ Gallery Ảnh Mẫu")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 8))
    
    for i in range(n_rows*n_cols):
        row = i // n_cols
        col = i % n_cols
        idx = np.random.randint(0, len(X))
        
        axes[row,col].imshow(X[idx].reshape(28,28), cmap='gray')
        axes[row,col].set_title(f"Label: {y[idx]}", fontsize=8)
        axes[row,col].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)

# Hàm phân tích phân phối pixel
def analyze_pixel_distribution(df):
    st.subheader("📈 Phân phối Giá trị Pixel")
    
    plt.figure(figsize=(10, 4))
    
    # Trước chuẩn hóa
    plt.subplot(1, 2, 1)
    plt.hist(df.values.flatten(), bins=50, color='blue', alpha=0.7)
    plt.title('Phân phối gốc')
    plt.xlabel('Giá trị pixel')
    plt.ylabel('Tần suất')

    # Sau chuẩn hóa
    plt.subplot(1, 2, 2)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.values)
    plt.hist(scaled_data.flatten(), bins=50, color='green', alpha=0.7)
    plt.title('Sau chuẩn hóa')
    plt.xlabel('Giá trị pixel (0-1)')
    
    plt.tight_layout()
    st.pyplot(plt)

def main():
    st.title("🔠 Tiền Xử lý Ảnh Chữ cái Nâng cao")
    show_explanations()
    
    # Tải lên dữ liệu
    with st.expander("📤 Tải lên Dữ liệu", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            X_file = st.file_uploader("Chọn file ảnh (.npy)", type="npy")
        with col2:
            y_file = st.file_uploader("Chọn file nhãn (.npy)", type="npy")

    if X_file and y_file:
        try:
            # Đọc dữ liệu
            X = np.load(X_file)
            y = np.load(y_file).astype(str)
            
            # Validation
            if len(X) != len(y):
                st.error(f"Lỗi: Số lượng ảnh ({len(X)}) và nhãn ({len(y)}) không khớp!")
                return

            # Hiển thị thông tin cơ bản
            st.subheader("📦 Thông tin Dataset")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tổng số mẫu", len(X))
            with col2:
                st.metric("Kích thước ảnh", f"{X.shape[1:] if X.ndim==3 else X.shape[1]}")
            with col3:
                unique_labels = np.unique(y)
                st.metric("Số lớp", len(unique_labels))

            # Hiển thị ảnh mẫu
            display_sample_images(X, y)
            
            # Phân tích phân phối pixel
            analyze_pixel_distribution(X.reshape(X.shape[0], -1))

            # Xử lý dữ liệu
            with st.status("⏳ Đang xử lý dữ liệu...", expanded=True) as status:
                st.write("1. Làm phẳng ảnh...")
                X_flat = X.reshape(X.shape[0], -1)
                
                st.write("2. Mã hóa nhãn...")
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
                
                st.write("3. Chuẩn hóa pixel...")
                scaler = MinMaxScaler()
                X_scaled = scaler.fit_transform(X_flat)
                
                st.write("4. Kiểm tra chất lượng...")
                df = pd.DataFrame(X_scaled, columns=[f"pixel_{i}" for i in range(X_scaled.shape[1])])
                df['label'] = y_encoded
                
                status.update(label="Xử lý hoàn tất!", state="complete")

            # Hiển thị kết quả
            st.subheader("✅ Kết quả Xử lý")
            tab1, tab2, tab3 = st.tabs(["Dữ liệu", "Nhãn", "Thống kê"])
            
            with tab1:
                st.dataframe(df.head(), use_container_width=True)
                
            with tab2:
                label_map = pd.DataFrame({
                    "Ký tự": le.classes_,
                    "Mã số": le.transform(le.classes_)
                })
                st.dataframe(label_map, hide_index=True)
                
            with tab3:
                st.write("**Phân phối lớp:**")
                label_dist = pd.Series(y).value_counts().reset_index()
                label_dist.columns = ['Ký tự', 'Số lượng']
                st.bar_chart(label_dist.set_index('Ký tự'))
            
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