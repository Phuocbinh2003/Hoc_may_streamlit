import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from PIL import Image

def drop(df):
    columns_to_drop = st.multiselect("Chọn cột muốn xóa", df.columns.tolist())
    if columns_to_drop:
        df.drop(columns=columns_to_drop, inplace=True)
        st.write("### Dữ liệu sau khi xóa cột:")
        st.dataframe(df.head())
    return df
def train_test_size(df):
    train_size = st.slider("Chọn % dữ liệu Train", 50, 90, 70)
    val_size = st.slider("Chọn % dữ liệu Validation", 0, 40, 15)
    test_size = 100 - train_size - val_size
    st.write(f"Tỷ lệ phân chia: Train={train_size}%, Validation={val_size}%, Test={test_size}%")
    # Chia dữ liệu: 70% train, 15% validation, 15% test
    train_data, temp_data = train_test_split(df, test_size=(100 - train_size)/100, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=test_size/(test_size + val_size), random_state=42)
    st.subheader("📊 Số lượng mẫu trong từng tập dữ liệu")
    summary_df = pd.DataFrame({
        "Tập dữ liệu": ["Train", "Validation", "Test"],
        "Số lượng mẫu": [train_df.shape[0], val_df.shape[0], test_df.shape[0]]
    })
    st.table(summary_df)
def xu_ly_gia_tri_thieu(df):
    st.subheader("Xử lý giá trị thiếu")
    
    if df.isnull().sum().sum() == 0:
        st.success("Không có giá trị thiếu trong dữ liệu!")
        return df

    # Chọn cột cần xử lý (chỉ hiển thị các cột có giá trị thiếu)
    missing_cols = df.columns[df.isnull().any()].tolist()
    selected_col = st.selectbox("Chọn cột chứa giá trị thiếu:", missing_cols)

    # Chọn phương pháp xử lý
    method = st.radio("Chọn phương pháp xử lý:", ["Thay thế bằng Mean", "Thay thế bằng Median", "Xóa giá trị thiếu"])

    # Xử lý dữ liệu
    if method == "Thay thế bằng Mean":
        df[selected_col].fillna(df[selected_col].mean(), inplace=True)
        st.success(f"Đã thay thế giá trị thiếu ở cột '{selected_col}' bằng Mean")
    elif method == "Thay thế bằng Median":
        df[selected_col].fillna(df[selected_col].median(), inplace=True)
        st.success(f"Đã thay thế giá trị thiếu ở cột '{selected_col}' bằng Median")
    elif method == "Xóa giá trị thiếu":
        df.dropna(subset=[selected_col], inplace=True)
        st.success(f"Đã xóa các dòng có giá trị thiếu trong cột '{selected_col}'")

    return df



def chuyen_doi_kieu_du_lieu(df):
    st.subheader("Chuyển đổi kiểu dữ liệu")

    # Chỉ lấy các cột kiểu object (chuỗi) để xử lý
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    if not categorical_cols:
        st.success("Không có thuộc tính dạng chuỗi cần chuyển đổi!")
        return df

    for col in categorical_cols:
        unique_values = df[col].unique()
        num_unique = len(unique_values)

        st.write(f"**Cột `{col}` có {num_unique} giá trị duy nhất:** {unique_values}")

        if num_unique > 10:
            st.warning(f"Cột `{col}` có hơn 10 giá trị duy nhất, có thể không phù hợp để chuyển đổi trực tiếp.")
        else:
            mapping_dict = {}
            for val in unique_values:
                new_val = st.text_input(f"Nhập giá trị thay thế cho `{val}` trong cột `{col}`", key=f"{col}_{val}")
                mapping_dict[val] = new_val

            # Chuyển đổi cột theo giá trị nhập vào
            df[col] = df[col].map(lambda x: mapping_dict.get(x, x))

            st.success(f"Đã chuyển đổi cột `{col}` với các giá trị: {mapping_dict}")

    return df
def chuan_hoa_du_lieu(df):
    st.subheader("Chuẩn hóa dữ liệu với StandardScaler")

    # Lọc các cột số để chuẩn hóa
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if not numerical_cols:
        st.success("Không có thuộc tính dạng số cần chuẩn hóa!")
        return df

    # Chọn cột số để chuẩn hóa
    selected_cols = st.multiselect("Chọn các cột số để chuẩn hóa", numerical_cols)

    if selected_cols:
        scaler = StandardScaler()
        df[selected_cols] = scaler.fit_transform(df[selected_cols])
        st.success(f"Đã chuẩn hóa các cột: {selected_cols}")

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
    drop(df)
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
    xu_ly_gia_tri_thieu(df)

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

    chuyen_doi_kieu_du_lieu(df)

    st.subheader("4️⃣ Chuẩn hóa dữ liệu số")
    st.write("""
        Các giá trị số có thể có khoảng giá trị khác nhau, làm ảnh hưởng đến mô hình. Ta sẽ chuẩn hóa "Age" và "Fare" về cùng một thang đo bằng StandardScaler.
        ```python
            scaler = StandardScaler()
            df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])

        ```
        """)

    
    chuan_hoa_du_lieu(df)
    
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
       
    train_test_size(df)
    

def tien_xu_ly_du_lieu():
    uploaded_file = st.file_uploader("📂 Chọn file dữ liệu (.csv hoặc .txt)", type=["csv", "txt"])

    
    df = pd.read_csv(uploaded_file, delimiter=",")
    hien_thi_ly_thuyet(df)
    if "Age" in df.columns:
        df["Age"].fillna(df["Age"].mean(), inplace=True)
        df['Age'] = df['Age'].astype(int)
        scaler = StandardScaler()
        df[['Age']] = scaler.fit_transform(df[['Age']])
                    
    if "Fare" in df.columns:
        df["Fare"].fillna(df["Fare"].median(), inplace=True)  # Điền giá trị trung vị
        df['Fare'] = df['Fare'].astype(int)
    if "Embarked" in df.columns:
        df.dropna(subset=['Embarked'], inplace=True)
        df['Embarked'] = df['Embarked'].map({'C': 1, 'S': 2,'Q': 3})
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
    
        


            
  

if __name__ == "__main__":
    tien_xu_ly_du_lieu(df)
