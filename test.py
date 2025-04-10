import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import zscore

# Hàm tiền xử lý dữ liệu từ file .npy
def tien_xu_ly_du_lieu_from_npy(X_file, y_file):
    # Tải dữ liệu từ các file .npy
    X = np.load(X_file, allow_pickle=True)
    y = np.load(y_file, allow_pickle=True)
    
    # Kiểm tra xem dữ liệu X có 3 chiều không (đối với hình ảnh)
    if X.ndim == 3:
        # Làm phẳng dữ liệu 3 chiều (mỗi hình ảnh trở thành một vector)
        X = X.reshape(X.shape[0], -1)  # Chuyển từ (10000, 28, 28) thành (10000, 784)
    
    # Chuyển dữ liệu NumPy thành DataFrame để dễ xử lý
    df = pd.DataFrame(X, columns=["Feature_" + str(i) for i in range(X.shape[1])])
    df['Target'] = y
    
    # Hiển thị thông tin dữ liệu gốc
    st.write("📊 **Dữ liệu gốc**:")
    st.write(df.head(10))

    # Kiểm tra các giá trị thiếu
    missing_values = df.isnull().sum()
    st.write("🔍 **Kiểm tra giá trị thiếu**:")
    st.write(missing_values)

    # Kiểm tra dữ liệu trùng lặp
    duplicate_count = df.duplicated().sum()
    st.write(f"🔁 **Số lượng dòng bị trùng lặp**: {duplicate_count}")

    # Kiểm tra outliers (Sử dụng Z-score)
    outlier_count = {
        col: (abs(zscore(df[col], nan_policy='omit')) > 3).sum()
        for col in df.select_dtypes(include=['number']).columns
    }
    st.write("🚨 **Outliers** (Z-score > 3):")
    st.write(outlier_count)

    # Xử lý các cột kiểu chữ (alphabet) bằng LabelEncoder
    label_encoder = LabelEncoder()
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = label_encoder.fit_transform(df[column])

    # Tiền xử lý các giá trị thiếu
    df['Target'] = df['Target'].fillna(df['Target'].mode()[0])  # Điền giá trị thiếu bằng giá trị mode
    df.dropna(inplace=True)  # Loại bỏ các dòng chứa giá trị thiếu nếu cần

    # Chuẩn hóa dữ liệu số
    scaler = StandardScaler()
    df[df.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(df.select_dtypes(include=[np.number]))

    # Hiển thị dữ liệu sau khi tiền xử lý
    st.write("✅ **Dữ liệu sau khi tiền xử lý**:")
    st.write(df.head(10))

    return df

# Hàm để hiển thị và tiền xử lý
def show_preprocessing_tab():
    st.title("🔍 Tiền xử lý Dữ liệu - Alphabet (từ .npy)")

    # Chọn tệp .npy
    X_file = st.file_uploader("📂 Tải lên tệp dữ liệu X (.npy)", type=["npy"])
    y_file = st.file_uploader("📂 Tải lên tệp dữ liệu y (.npy)", type=["npy"])
    
    # Nếu người dùng tải lên cả X và y, thực hiện tiền xử lý
    if X_file is not None and y_file is not None:
        # Lưu tệp tải lên tạm thời
        with open("/mnt/data/X_data.npy", "wb") as f:
            f.write(X_file.getbuffer())
        with open("/mnt/data/y_data.npy", "wb") as f:
            f.write(y_file.getbuffer())

        # Gọi hàm tiền xử lý dữ liệu từ các tệp .npy
        df = tien_xu_ly_du_lieu_from_npy("/mnt/data/X_data.npy", "/mnt/data/y_data.npy")
    else:
        st.warning("⚠️ Vui lòng tải lên cả hai tệp dữ liệu X và y!")

# Gọi hàm trong Streamlit
if __name__ == "__main__":
    show_preprocessing_tab()
