import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import zscore

# Hàm tiền xử lý dữ liệu trực tiếp từ mảng NumPy
def tien_xu_ly_du_lieu(X, y):
    # Kiểm tra và làm phẳng dữ liệu ảnh
    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)
    
    # Tạo DataFrame
    df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(X.shape[1])])
    df['Target'] = y
    
    # Hiển thị thông tin
    st.write("📊 **Dữ liệu gốc:**")
    st.write(df.head(10))

    # Kiểm tra giá trị thiếu
    st.write("🔍 **Giá trị thiếu:**")
    st.write(df.isnull().sum())

    # Kiểm tra trùng lặp
    st.write(f"🔁 **Dòng trùng lặp:** {df.duplicated().sum()}")

    # Phát hiện outliers
    numeric_cols = df.select_dtypes(include='number').columns.drop('Target', errors='ignore')
    outliers = {col: (np.abs(zscore(df[col], nan_policy='omit')) > 3).sum() for col in numeric_cols}
    st.write("🚨 **Outliers (Z-score > 3):**")
    st.write(outliers)

    # Xử lý dữ liệu phân loại
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Xử lý giá trị thiếu
    df['Target'] = df['Target'].fillna(df['Target'].mode()[0])
    df.dropna(inplace=True)

    # Chuẩn hóa dữ liệu (không bao gồm target)
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    st.write("✅ **Dữ liệu sau xử lý:**")
    st.write(df.head(10))
    
    return df

# Giao diện chính
def show_preprocessing_tab():
    st.title("🔍 Tiền xử lý Dữ liệu - Alphabet (từ .npy)")
    
    # Tải lên file
    X_file = st.file_uploader("📂 Tải lên file X (.npy)", type="npy")
    y_file = st.file_uploader("📂 Tải lên file y (.npy)", type="npy")
    
    if X_file and y_file:
        try:
            # Đọc trực tiếp từ file upload
            X = np.load(X_file, allow_pickle=True)
            y = np.load(y_file, allow_pickle=True)
            
            # Xử lý và hiển thị
            if y.ndim > 1:
                y = y.squeeze()
            df = tien_xu_ly_du_lieu(X, y)
            
            # Thêm tính năng download
            st.download_button(
                label="📥 Tải xuống dữ liệu đã xử lý",
                data=df.to_csv().encode(),
                file_name="processed_data.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Lỗi: {str(e)}")
    else:
        st.warning("⚠️ Vui lòng tải lên cả 2 file!")

if __name__ == "__main__":
    show_preprocessing_tab()