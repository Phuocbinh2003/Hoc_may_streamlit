import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import zscore

# Hàm tiền xử lý dữ liệu
def tien_xu_ly_du_lieu(updates_file=None):
    if updates_file is not None:
        # Đọc file upload từ người dùng
        df = pd.read_csv(updates_file)
    else:
        # Giả sử bạn có dữ liệu mặc định
        X = np.load('/mnt/data/alphabet_X.npy', allow_pickle=True)
        y = np.load('/mnt/data/alphabet_y.npy', allow_pickle=True)
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
    st.title("🔍 Tiền xử lý Dữ liệu - Alphabet")

    # Upload file dữ liệu
    uploaded_file = st.file_uploader("📂 Chọn file dữ liệu (.csv hoặc .txt)", type=["csv", "txt"])
    if uploaded_file is not None:
        df = tien_xu_ly_du_lieu(uploaded_file)
    else:
        st.warning("⚠️ Vui lòng tải lên tệp dữ liệu để tiến hành tiền xử lý!")

# Gọi hàm trong Streamlit
if __name__ == "__main__":
    show_preprocessing_tab()
