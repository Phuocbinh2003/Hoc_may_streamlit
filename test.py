import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.stats import zscore
import matplotlib.pyplot as plt

def preprocess_alphabet_data(X, y):
    # Tiền xử lý cho dữ liệu ảnh chữ cái
    st.write("📝 **Bước 1: Làm phẳng dữ liệu ảnh**")
    if X.ndim == 3:
        # Làm phẳng ảnh 28x28 thành một vector có 784 pixel
        X = X.reshape(X.shape[0], -1)  # Flatten ảnh 28x28 -> 784 pixels
        st.write(f"Đã làm phẳng dữ liệu từ kích thước {X.shape[1]}x{X.shape[2]} thành {X.shape[1]}")

    # Tạo DataFrame từ dữ liệu X
    st.write("📝 **Bước 2: Tạo DataFrame từ dữ liệu**")
    df = pd.DataFrame(X, columns=[f"pixel_{i}" for i in range(X.shape[1])])
    df['label'] = y  # Giả sử nhãn là các chữ cái A-Z
    
    # Hiển thị thống kê cơ bản về dữ liệu
    st.write("📊 **Thông tin dataset**:")
    st.write(f"- Số lượng mẫu: {len(df)}")
    st.write(f"- Số lớp: {len(np.unique(y))}")

    # Hiển thị ví dụ ảnh ngẫu nhiên
    st.write("🖼️ **Ví dụ dữ liệu ảnh**:")
    sample_idx = np.random.randint(0, len(X))  # Chọn một chỉ số ngẫu nhiên
    plt.imshow(X[sample_idx].reshape(28, 28), cmap='gray')  # Hiển thị ảnh dưới dạng 28x28
    plt.axis('off')  # Ẩn trục
    st.pyplot(plt)
    
    # Xử lý nhãn chữ cái (chuyển thành số)
    st.write("📝 **Bước 3: Mã hóa nhãn chữ cái**")
    le = LabelEncoder()  # Tạo đối tượng LabelEncoder
    df['label'] = le.fit_transform(df['label'])  # Chuyển đổi nhãn thành số
    
    # Lưu ánh xạ nhãn từ chữ cái thành số
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    st.write("🔡 **Ánh xạ nhãn**:", label_mapping)

    # Chuẩn hóa giá trị pixel từ [0, 255] về [0, 1]
    st.write("📝 **Bước 4: Chuẩn hóa giá trị pixel**")
    scaler = MinMaxScaler()
    pixel_columns = [col for col in df.columns if col.startswith('pixel')]
    df[pixel_columns] = scaler.fit_transform(df[pixel_columns])  # Chuẩn hóa tất cả cột pixel
    
    # Phân tích thống kê về các pixel
    st.write("🔍 **Phân tích pixel**:")
    pixel_stats = df[pixel_columns].agg(['mean', 'std', 'min', 'max'])
    st.write(pixel_stats)
    
    # Phát hiện và loại bỏ các ảnh hỏng (nếu có)
    st.write("📝 **Bước 5: Phát hiện và loại bỏ ảnh hỏng**")
    corrupted_images = df[(df[pixel_columns] < 0).any(axis=1) | (df[pixel_columns] > 1).any(axis=1)]
    if not corrupted_images.empty:
        st.warning(f"⚠️ Phát hiện {len(corrupted_images)} ảnh hỏng, đang loại bỏ...")
        df = df.drop(corrupted_images.index)  # Loại bỏ các ảnh hỏng khỏi DataFrame
    
    return df, le

def main():
    st.title("🎯 Tiền xử lý Ảnh Chữ cái")

    # Tải lên dữ liệu
    col1, col2 = st.columns(2)
    with col1:
        X_file = st.file_uploader("Tải lên file ảnh (.npy)", type="npy")
    with col2:
        y_file = st.file_uploader("Tải lên file nhãn (.npy)", type="npy")
    
    if X_file and y_file:
        try:
            X = np.load(X_file)  # Đọc dữ liệu ảnh từ file
            y = np.load(y_file)  # Đọc nhãn từ file
            
            # Kiểm tra kích thước dữ liệu
            if len(X) != len(y):
                st.error("Lỗi: Số lượng ảnh và nhãn không khớp!")
                return
                
            # Xử lý dữ liệu
            df, label_encoder = preprocess_alphabet_data(X, y)
            
            # Hiển thị kết quả sau khi xử lý
            st.write("✅ **Dữ liệu đã xử lý**:")
            st.dataframe(df.head())  # Hiển thị 5 dòng đầu tiên của DataFrame
            
            # Tải xuống dữ liệu đã xử lý
            st.download_button(
                label="📥 Tải xuống dữ liệu đã xử lý",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name="alphabet_processed.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Lỗi xử lý: {str(e)}")
    else:
        st.info("ℹ️ Vui lòng tải lên cả file ảnh và file nhãn")

if __name__ == "__main__":
    main()
