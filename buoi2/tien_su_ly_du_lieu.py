import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from PIL import Image

def drop(df):
    st.subheader("🗑️ Xóa cột dữ liệu")
    
    if "df" not in st.session_state:
        st.session_state.df = df  # Lưu vào session_state nếu chưa có

    df = st.session_state.df
    columns_to_drop = st.multiselect("📌 Chọn cột muốn xóa:", df.columns.tolist())

    if st.button("🚀 Xóa cột đã chọn"):
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)  # Tạo bản sao thay vì inplace=True
            st.session_state.df = df  # Cập nhật session_state
            st.success(f"✅ Đã xóa cột: {', '.join(columns_to_drop)}")
            st.dataframe(df.head())
        else:
            st.warning("⚠️ Vui lòng chọn ít nhất một cột để xóa!")

    return df
def train_test_size(df):
    st.subheader("📊 Chia dữ liệu Train - Validation - Test")

    train_size = st.slider("Chọn % dữ liệu Train", 50, 90, 70)
    val_size = st.slider("Chọn % dữ liệu Validation", 0, 40, 15)
    test_size = 100 - train_size - val_size

    st.write(f"📌 **Tỷ lệ phân chia:** Train={train_size}%, Validation={val_size}%, Test={test_size}%")

    # Chia dữ liệu
    train_df, temp_df = train_test_split(df, test_size=(100 - train_size) / 100, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=test_size / (test_size + val_size), random_state=42)

    st.session_state.train_df = train_df
    st.session_state.val_df = val_df
    st.session_state.test_df = test_df

    # Hiển thị thông tin số lượng mẫu
    summary_df = pd.DataFrame({
        "Tập dữ liệu": ["Train", "Validation", "Test"],
        "Số lượng mẫu": [train_df.shape[0], val_df.shape[0], test_df.shape[0]]
    })
    st.table(summary_df)

    
def xu_ly_gia_tri_thieu(df):
    st.subheader("⚡ Xử lý giá trị thiếu")

    if "df" not in st.session_state:
        st.session_state.df = df.copy()

    df = st.session_state.df

    # Tìm cột có giá trị thiếu
    missing_cols = df.columns[df.isnull().any()].tolist()
    if not missing_cols:
        st.success("✅ Dữ liệu không có giá trị thiếu!")
        return df

    selected_col = st.selectbox("📌 Chọn cột chứa giá trị thiếu:", missing_cols)
    method = st.radio("🔧 Chọn phương pháp xử lý:", ["Thay thế bằng Mean", "Thay thế bằng Median", "Xóa giá trị thiếu"])

    if st.button("🚀 Xử lý giá trị thiếu"):
        if method == "Thay thế bằng Mean":
            df[selected_col] = df[selected_col].fillna(df[selected_col].mean())
        elif method == "Thay thế bằng Median":
            df[selected_col] = df[selected_col].fillna(df[selected_col].median())
        elif method == "Xóa giá trị thiếu":
            df = df.dropna(subset=[selected_col])

        st.session_state.df = df
        st.success(f"✅ Đã xử lý giá trị thiếu trong cột `{selected_col}`")

    st.dataframe(df.head())
    return df




def chuyen_doi_kieu_du_lieu(df):
    st.subheader("🔄 Chuyển đổi kiểu dữ liệu")

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if not categorical_cols:
        st.success("✅ Không có cột dạng chuỗi cần chuyển đổi!")
        return df

    selected_col = st.selectbox("📌 Chọn cột để chuyển đổi:", categorical_cols)
    unique_values = df[selected_col].unique()
    
    mapping_dict = {}
    if len(unique_values) <10:
        for val in unique_values:
            new_val = st.text_input(f"🔄 Nhập giá trị thay thế cho `{val}`:", key=f"{selected_col}_{val}")
            mapping_dict[val] = new_val

        if st.button("🚀 Chuyển đổi dữ liệu"):
            df[selected_col] = df[selected_col].map(lambda x: mapping_dict.get(x, x))
            st.session_state.df = df
            st.success(f"✅ Đã chuyển đổi cột `{selected_col}`")
    
    st.dataframe(df.head())
    return df
def chuan_hoa_du_lieu(df):
    st.subheader("📊 Chuẩn hóa dữ liệu với StandardScaler")

    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    if not numerical_cols:
        st.success("✅ Không có thuộc tính dạng số cần chuẩn hóa!")
        return df

    selected_cols = st.multiselect("📌 Chọn các cột số để chuẩn hóa:", numerical_cols)

    if st.button("🚀 Thực hiện chuẩn hóa"):
        if selected_cols:
            scaler = StandardScaler()
            df[selected_cols] = scaler.fit_transform(df[selected_cols])
            st.session_state.df = df
            st.success(f"✅ Đã chuẩn hóa các cột: {', '.join(selected_cols)}")

    st.dataframe(df.head())
    return df

def hien_thi_ly_thuyet(df):
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
   
    
    st.title("🔍 Tiền xử lý dữ liệu")

    # Hiển thị dữ liệu gốc
    
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
    df1=drop(df)
    
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
    df=xu_ly_gia_tri_thieu(df1)

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

    df=chuyen_doi_kieu_du_lieu(df)

    st.subheader("4️⃣ Chuẩn hóa dữ liệu số")
    st.write("""
        Các giá trị số có thể có khoảng giá trị khác nhau, làm ảnh hưởng đến mô hình. Ta sẽ chuẩn hóa "Age" và "Fare" về cùng một thang đo bằng StandardScaler.
        ```python
            scaler = StandardScaler()
            df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])

        ```
        """)

    
    df=chuan_hoa_du_lieu(df)
    
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
       
    df=train_test_size(df)
    

def tien_xu_ly_du_lieu():
    uploaded_file = st.file_uploader("📂 Chọn file dữ liệu (.csv hoặc .txt)", type=["csv", "txt"])
    if uploaded_file is not None:  # Kiểm tra xem file đã được tải lên chưa
        try:
            df = pd.read_csv(uploaded_file, delimiter=",")
            hien_thi_ly_thuyet(df)
        except Exception as e:
            st.error(f"❌ Lỗi khi đọc file: {e}")
  
    
        


            
  

if __name__ == "__main__":
    tien_xu_ly_du_lieu()
