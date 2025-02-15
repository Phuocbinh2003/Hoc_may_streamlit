import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def tien_xu_ly_du_lieu():
    st.title("📊 Xử lý Dữ liệu & Chia Train/Test/Validation")

    # Upload file thay vì dùng đường dẫn cố định
    uploaded_file = st.file_uploader("📂 Chọn file dữ liệu (.csv hoặc .txt)", type=["csv", "txt"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, delimiter=",")  # Điều chỉnh delimiter nếu cần

            # 1️⃣ Hiển thị 10 dòng đầu tiên
            st.subheader("📌 10 dòng đầu của dữ liệu gốc")
            st.write(df.head(10))

            # 2️⃣ Kiểm tra lỗi dữ liệu
            st.subheader("🚨 Kiểm tra lỗi dữ liệu")
            missing_values = df.isnull().sum()
            error_report = pd.DataFrame({'Cột': df.columns, 'Giá trị thiếu': missing_values})
            st.table(error_report)

            # 3️⃣ Xử lý lỗi dữ liệu
            if "Embarked" in df.columns:
                df.dropna(subset=['Embarked'], inplace=True)

            if "Age" in df.columns:
                df['Age'].fillna(df['Age'].mean(), inplace=True)
                df['Age'] = df['Age'].astype(int)  # Đảm bảo Age là số nguyên

            if "Cabin" in df.columns:
                df['Cabin'].fillna('Unknown', inplace=True)

            if "Pclass" in df.columns:
                df['Pclass'] = df['Pclass'].astype('category')

            if "Sex" in df.columns:
                df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

            if "Fare" in df.columns and df['Fare'].nunique() > 1:
                scaler = StandardScaler()
                df[['Fare']] = scaler.fit_transform(df[['Fare']])

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

        except Exception as e:
            st.error(f"⚠️ Lỗi khi xử lý dữ liệu: {e}")

if __name__ == "__main__":
    tien_xu_ly_du_lieu()
