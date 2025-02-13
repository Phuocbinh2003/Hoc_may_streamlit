import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split

# Tiêu đề ứng dụng
st.title("📊 Xử lý Dữ liệu & Chia Train/Test/Validation")

# Đường dẫn file data.txt
file_path = "data.txt"

# Kiểm tra file tồn tại
try:
    df = pd.read_csv(file_path, delimiter=",")  # Điều chỉnh delimiter nếu cần

    # 1️⃣ Hiển thị 10 dòng đầu tiên
    st.subheader("📌 10 dòng đầu của dữ liệu gốc")
    st.write(df.head(10))

    # 2️⃣ Kiểm tra lỗi dữ liệu
    st.subheader("🚨 Kiểm tra lỗi dữ liệu")
    missing_values = df.isnull().sum()
    error_report = pd.DataFrame({'Cột': df.columns, 'Giá trị thiếu': missing_values})
    st.table(error_report)

    # 3️⃣ Xử lý lỗi dữ liệu
    df.fillna(df.mean(), inplace=True)  # Điền giá trị thiếu bằng trung bình số

    # Hiển thị dữ liệu sau khi xử lý
    st.subheader("✅ Dữ liệu sau xử lý")
    st.write(df.head(10))

    # 4️⃣ Chia dữ liệu: 70% train, 15% validation, 15% test
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # 5️⃣ In số lượng mẫu
    st.subheader("📊 Số lượng mẫu trong từng tập dữ liệu")
    summary_df = pd.DataFrame({
        "Tập dữ liệu": ["Train", "Validation", "Test"],
        "Số lượng mẫu": [train_df.shape[0], val_df.shape[0], test_df.shape[0]]
    })
    st.table(summary_df)

except FileNotFoundError:
    st.error("⚠️ Không tìm thấy file `data.txt`. Vui lòng kiểm tra đường dẫn!")
except Exception as e:
    st.error(f"⚠️ Lỗi khi xử lý dữ liệu: {e}")
