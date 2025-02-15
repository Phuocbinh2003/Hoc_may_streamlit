import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
def hien_thi_ly_thuyet():
    st.title("📊 Xử lý Dữ liệu & Chia Train/Test/Validation")

    st.subheader("1️⃣ Giới thiệu về Tiền xử lý Dữ liệu")
    st.write("""
    Tiền xử lý dữ liệu là một bước quan trọng trong phân tích dữ liệu và học máy. Nó giúp dữ liệu trở nên sạch và phù hợp hơn để sử dụng. 
    Một số vấn đề phổ biến trong dữ liệu:
    - **Giá trị rỗng** (NaN, None)
    - **Định dạng không đồng nhất** (chuỗi, số, ngày tháng)
    - **Dữ liệu lỗi** (giá trị âm, không hợp lệ)
    - **Dữ liệu trùng lặp**
    """)

    st.header("⚙️ Các bước chính trong tiền xử lý dữ liệu")
    st.image("buoi2\img1.png", caption="Làm sạch dữ liệu", use_column_width=True)

    st.subheader("1️⃣ Xử lý giá trị rỗng")
    st.write("""
    Dữ liệu thường có những ô bị thiếu thông tin (NaN), có thể xử lý theo nhiều cách:
    - **Xóa dòng/cột chứa giá trị rỗng**: Dùng `dropna()`
    - **Điền giá trị mặc định**: Dùng `fillna()`
    - **Dùng trung bình, trung vị, hoặc giá trị phổ biến nhất**:  
      ```python
      df['column'].fillna(df['column'].mean())
      ```
    """)

    st.subheader("2️⃣ Chuyển đổi kiểu dữ liệu")
    st.write("""
    Một số cột có thể cần chuyển đổi kiểu dữ liệu:
    - **Chuyển cột số thành dạng phân loại**:  
      ```python
      df['Pclass'] = df['Pclass'].astype('category')
      ```
    - **Mã hóa biến phân loại (ví dụ: giới tính)**:  
      ```python
      df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
      ```
    """)

    st.subheader("3️⃣ Chuẩn hóa dữ liệu")
    st.write("""
    Để đảm bảo dữ liệu có cùng khoảng giá trị, ta chuẩn hóa bằng StandardScaler:
    ```python
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[['Fare']] = scaler.fit_transform(df[['Fare']])
    ```
    """)
    

    st.subheader("4️⃣ Xử lý dữ liệu trùng lặp")
    st.write("""
    Dữ liệu có thể bị trùng, gây ảnh hưởng đến kết quả phân tích:
    - **Kiểm tra dữ liệu trùng lặp**:  
      ```python
      df.duplicated().sum()
      ```
    - **Xóa dữ liệu trùng lặp**:  
      ```python
      df = df.drop_duplicates()
      ```
    """)

    st.subheader("5️⃣ Chia dữ liệu thành Train - Validation - Test")
    st.write("""
    Dữ liệu được chia thành:
    - **Tập Train (70%)**: Dùng để huấn luyện mô hình.
    - **Tập Validation (15%)**: Dùng để điều chỉnh mô hình.
    - **Tập Test (15%)**: Kiểm tra mô hình với dữ liệu mới.
    """)

def tien_xu_ly_du_lieu():
    # Upload file
    uploaded_file = st.file_uploader("📂 Chọn file dữ liệu (.csv hoặc .txt)", type=["csv", "txt"])

    if uploaded_file is None:
        hien_thi_ly_thuyet()  # Chỉ hiển thị lý thuyết nếu chưa có file tải lên
    
    else:
        try:
            df = pd.read_csv(uploaded_file, delimiter=",")  # Điều chỉnh delimiter nếu cần

            # Hiển thị dữ liệu ban đầu
            st.subheader("📌 10 dòng đầu của dữ liệu gốc")
            st.write(df.head(10))

            # Kiểm tra lỗi dữ liệu
            st.subheader("🚨 Kiểm tra lỗi dữ liệu")

            # Kiểm tra giá trị thiếu
            missing_values = df.isnull().sum()

            # Kiểm tra dữ liệu trùng lặp
            duplicate_count = df.duplicated().sum()

            # Kiểm tra giá trị âm (chỉ hiển thị nếu > 0)
            invalid_values = {
                col: (df[col] < 0).sum() for col in df.select_dtypes(include=['number']).columns
            }
            invalid_values = {k: v for k, v in invalid_values.items() if v > 0}  # Bỏ giá trị âm = 0

            # Kiểm tra giá trị quá lớn (outlier) bằng Z-score
            outlier_count = {
                col: (abs(zscore(df[col], nan_policy='omit')) > 3).sum()
                for col in df.select_dtypes(include=['number']).columns
            }

            # Tạo báo cáo lỗi
            error_report = pd.DataFrame({
                'Cột': df.columns,
                'Giá trị thiếu': missing_values,

                'Outlier': [outlier_count.get(col, 0) for col in df.columns]
            })

            # Hiển thị báo cáo lỗi
            st.table(error_report)

            # Hiển thị số lượng dữ liệu trùng lặp
            st.write(f"🔁 **Số lượng dòng bị trùng lặp:** {duplicate_count}")

            # Xử lý lỗi dữ liệu
            if "Embarked" in df.columns:
                df.dropna(subset=['Embarked'], inplace=True)

            if "Age" in df.columns:
                df['Age'].fillna(df['Age'].mean(), inplace=True)
                df['Age'] = df['Age'].astype(int)

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

            # Chia dữ liệu: 70% train, 15% validation, 15% test
            train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
            val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

            # Hiển thị số lượng mẫu
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
