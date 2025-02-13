import streamlit as st
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Tiêu đề ứng dụng
st.title("📊 Xử lý Dữ liệu & Chia Train/Test/Validation")

# Nhập đường dẫn thư mục chứa file CSV
folder_path = hoc_may_data_b2.txt

# Kiểm tra thư mục hợp lệ
if os.path.isdir(folder_path):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    selected_file = st.selectbox("📌 Chọn file dữ liệu:", csv_files) if csv_files else None

    if selected_file:
        file_path = os.path.join(folder_path, selected_file)
        df = pd.read_csv(file_path)

        # 1️⃣ Hiển thị 10 dòng đầu tiên
        st.subheader("📌 10 dòng đầu của dữ liệu gốc")
        st.write(df.head(10))

        # 2️⃣ Kiểm tra lỗi dữ liệu
        st.subheader("🚨 Kiểm tra lỗi dữ liệu")
        missing_values = df.isnull().sum()
        invalid_values = (df == '').sum()
        error_report = pd.DataFrame({'Cột': df.columns, 'Giá trị thiếu': missing_values, 'Giá trị không hợp lệ': invalid_values})
        st.table(error_report)

        # 3️⃣ Xử lý lỗi dữ liệu
        st.subheader("🔧 Xử lý lỗi dữ liệu")
        df['Age'].fillna(df['Age'].mean(), inplace=True)  # Điền giá trị thiếu bằng trung bình
        df['Cabin'].fillna('Unknown', inplace=True)  # Điền Cabin thiếu bằng 'Unknown'
        df.dropna(subset=['Embarked'], inplace=True)  # Xóa dòng thiếu Embarked
        df['Age'] = df['Age'].astype(int)
        df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})  # Chuyển giới tính thành số
        df['Pclass'] = df['Pclass'].astype('category')

        if 'Fare' in df.columns and df['Fare'].nunique() > 1:
            scaler = StandardScaler()
            df['Fare'] = scaler.fit_transform(df[['Fare']])  # Chuẩn hóa giá vé

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

        # Cho phép tải xuống dữ liệu đã chia
        st.subheader("📥 Tải xuống dữ liệu:")
        for name, data in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(label=f"📂 Tải {name} set", data=csv, file_name=f"{name.lower()}_data.csv", mime="text/csv")

else:
    st.error("⚠️ Đường dẫn thư mục không hợp lệ! Vui lòng nhập lại.")
