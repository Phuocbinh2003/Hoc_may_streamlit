import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from PIL import Image
def hien_thi_ly_thuyet():
    uploaded_file = "buoi2/data.txt"
    try:
        df = pd.read_csv(uploaded_file, delimiter=",")
    except FileNotFoundError:
        st.error("❌ Không tìm thấy tệp dữ liệu. Vui lòng kiểm tra lại đường dẫn.")
        st.stop()
    
    st.title("🔍 Tiền xử lý dữ liệu")

    # Hiển thị dữ liệu gốc
    st.subheader("📌 10 dòng đầu của dữ liệu gốc")
    st.write(df.head(10))
    st.header("⚙️ Các bước chính trong tiền xử lý dữ liệu")
    st.subheader("1️⃣ Loại bỏ các cột không cần thiết")
    st.write("""
        Một số cột trong dữ liệu có thể không ảnh hưởng đến kết quả dự đoán hoặc chứa quá nhiều giá trị thiếu. Chúng ta sẽ loại bỏ các cột như:
        - **Cabin**: Cột này có quá nhiều giá trị bị thiếu.
        - **Ticket**: Mã vé không mang nhiều thông tin hữu ích.
        - **Name**:  Không cần thiết cho bài toán dự đoán sống sót.
        ```python
            columns_to_drop = ["Cabin", "Ticket", "Name"]  
            df.drop(columns=columns_to_drop, inplace=True)
        ```
        """)
    columns_to_drop = ["Cabin", "Ticket", "Name"]  # Cột không cần thiết
    df.drop(columns=columns_to_drop, inplace=True)  # Loại bỏ cột

    st.subheader("2️⃣ Xử lý giá trị thiếu")
    st.write("""
        Dữ liệu thực tế thường có giá trị bị thiếu. Ta cần xử lý để tránh ảnh hưởng đến mô hình.
        - **Cột "Age"**: Điền giá trị trung bình vì đây là dữ liệu số.
        - **Cột "Fare"**: Điền giá trị trung vị để giảm ảnh hưởng của ngoại lai.
        - **Cột "Embarked"**:   Xóa các dòng bị thiếu vì số lượng ít.
        ```python
            df["Age"].fillna(df["Age"].mean(), inplace=True)  # Điền giá trị trung bình cho "Age"
            df["Fare"].fillna(df["Fare"].median(), inplace=True)  # Điền giá trị trung vị cho "Fare"
            df.dropna(subset=["Embarked"], inplace=True)  # Xóa dòng thiếu "Embarked"

        ```
        """)
    df["Age"].fillna(df["Age"].mean(), inplace=True)  # Điền giá trị trung bình
    df["Fare"].fillna(df["Fare"].median(), inplace=True)  # Điền giá trị trung vị
    df.dropna(subset=["Embarked"], inplace=True)  # Xóa dòng thiếu Embarked

    st.subheader("3️⃣ Chuyển đổi kiểu dữ liệu")
    st.write("""
        Trong dữ liệu, có một số cột chứa giá trị dạng chữ (category). Ta cần chuyển đổi thành dạng số để mô hình có thể xử lý.
        - **Cột "Sex"**: Chuyển thành 1 (Nam), 0 (Nữ).
        - **Cột "Embarked"**:   Dùng One-Hot Encoding để tạo các cột mới cho từng giá trị ("S", "C", "Q").
        ```python
            df["Sex"] = df["Sex"].map({"male": 1, "female": 0})  # Mã hóa giới tính
            df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)  # One-Hot Encoding


        ```
        """)
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0})  # Mã hóa giới tính
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)  # One-Hot Encoding


    st.subheader("4️⃣ Chuẩn hóa dữ liệu số")
    st.write("""
        Các giá trị số có thể có khoảng giá trị khác nhau, làm ảnh hưởng đến mô hình. Ta sẽ chuẩn hóa "Age" và "Fare" về cùng một thang đo bằng StandardScaler.
        ```python
            scaler = StandardScaler()
            df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])

        ```
        """)
    scaler = StandardScaler()
    df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])

    st.write("Dữ liệu sau khi xử lý:")
    st.write(df.head(10))

    st.subheader("5️⃣ Chia dữ liệu thành tập Train, Validation, và Test")
    st.write("""
        Dữ liệu được chia thành ba phần để đảm bảo mô hình tổng quát tốt:
        - **70%**: để train mô hình.
        - **15%**: để validation, dùng để điều chỉnh tham số.
        - **15%"**:   để test, đánh giá hiệu suất thực tế.
        ```python
            # Chia dữ liệu theo tỷ lệ 70% và 30% (train - temp)
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

            # Chia tiếp 30% thành 15% validation và 15% test
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        ```
        """)
    X = df.drop(columns=["Survived"])  # Biến đầu vào
    y = df["Survived"]  # Nhãn
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

    # Chia tiếp 30% thành 15% validation và 15% test
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    st.write("📌 Số lượng mẫu trong từng tập dữ liệu:")
    st.write(f"👉 Train: {X_train.shape[0]} mẫu")
    st.write(f"👉 Validation: {X_val.shape[0]} mẫu")
    st.write(f"👉 Test: {X_test.shape[0]} mẫu")
    
    

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
