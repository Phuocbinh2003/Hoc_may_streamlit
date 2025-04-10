import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.stats import zscore
import matplotlib.pyplot as plt

def preprocess_alphabet_data(X, y):
    # Tiền xử lý cho dữ liệu ảnh chữ cái
    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)  # Flatten ảnh 28x28 -> 784 pixels
    
    # Tạo DataFrame
    df = pd.DataFrame(X, columns=[f"pixel_{i}" for i in range(X.shape[1])])
    df['label'] = y  # Giả sử nhãn là các chữ cái A-Z
    
    # Hiển thị thống kê
    st.write("📊 **Thông tin dataset**:")
    st.write(f"- Số lượng mẫu: {len(df)}")
    st.write(f"- Số lớp: {len(np.unique(y))}")
    
    # Hiển thị ví dụ ảnh
    st.write("🖼️ **Ví dụ dữ liệu ảnh**:")
    sample_idx = np.random.randint(0, len(X))
    plt.imshow(X[sample_idx].reshape(28, 28), cmap='gray')
    plt.axis('off')
    st.pyplot(plt)
    
    # Xử lý nhãn chữ cái
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])
    
    # Lưu ánh xạ nhãn
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    st.write("🔡 **Ánh xạ nhãn**:", label_mapping)

    # Chuẩn hóa pixel values về [0, 1]
    scaler = MinMaxScaler()
    pixel_columns = [col for col in df.columns if col.startswith('pixel')]
    df[pixel_columns] = scaler.fit_transform(df[pixel_columns])
    
    # Phát hiện outliers (đặc thù ảnh)
    st.write("🔍 **Phân tích pixel**:")
    pixel_stats = df[pixel_columns].agg(['mean', 'std', 'min', 'max'])
    st.write(pixel_stats)
    
    # Loại bỏ ảnh hỏng (nếu có)
    corrupted_images = df[(df[pixel_columns] < 0).any(axis=1) | (df[pixel_columns] > 1).any(axis=1)]
    if not corrupted_images.empty:
        st.warning(f"⚠️ Phát hiện {len(corrupted_images)} ảnh hỏng, đang loại bỏ...")
        df = df.drop(corrupted_images.index)
    
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
            X = np.load(X_file)
            y = np.load(y_file)
            
            # Kiểm tra kích thước
            if len(X) != len(y):
                st.error("Lỗi: Số lượng ảnh và nhãn không khớp!")
                return
                
            # Xử lý dữ liệu
            df, label_encoder = preprocess_alphabet_data(X, y)
            
            # Hiển thị kết quả
            st.write("✅ **Dữ liệu đã xử lý**:")
            st.dataframe(df.head())
            
            # Tải xuống
            st.download_button(
                label="📥 Tải xuống dữ liệu",
                data=df.to_csv().encode(),
                file_name="alphabet_processed.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Lỗi xử lý: {str(e)}")
    else:
        st.info("ℹ️ Vui lòng tải lên cả file ảnh và file nhãn")

if __name__ == "__main__":
    main()