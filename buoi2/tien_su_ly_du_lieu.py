import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import zscore
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error
import mlflow
import io
from sklearn.model_selection import KFold

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

def choose_label(df):
    st.subheader("🎯 Chọn cột dự đoán (label)")
    
    if "target_column" not in st.session_state:
        st.session_state.target_column = None
    
    selected_label = st.selectbox("📌 Chọn cột dự đoán", df.columns, 
                                  index=df.columns.get_loc(st.session_state.target_column) if st.session_state.target_column else 0)

    if st.button("✅ Xác nhận Label"):
        st.session_state.target_column = selected_label
        st.success(f"✅ Đã chọn cột: **{selected_label}**")
        
    
def train_test_size(df):
    st.subheader("📊 Chia dữ liệu Train - Validation - Test")
    
    # Kiểm tra đã chọn label chưa
    if "target_column" not in st.session_state or st.session_state.target_column is None:
        st.warning("⚠ Vui lòng chọn cột dự đoán trước!")
        choose_label(df)  # Gọi hàm chọn label trước
        return None, None, None, None, None, None

    target_column = st.session_state.target_column

    if st.button("✅ Chia dữ liệu"):
        X = df.drop(columns=[target_column])
        y = df[target_column]
        st.write(y)
        
        # Chọn tỷ lệ tập Test và Validation
        test_size = st.slider("📌 Chọn % dữ liệu Test", 10, 50, 20)
        remaining_size = 100 - test_size
        val_size = st.slider("📌 Chọn % dữ liệu Validation (trong phần Train)", 0, 50, 15)

        # Chia dữ liệu
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, stratify=y, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size / (100 - test_size), stratify=y_train, random_state=42)

        # Lưu vào session_state
        st.session_state.X_train = X_train
        st.session_state.X_val = X_val
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_val = y_val
        st.session_state.y_test = y_test

        # Hiển thị thông tin
        summary_df = pd.DataFrame({
            "Tập dữ liệu": ["Train", "Validation", "Test"],
            "Số lượng mẫu": [X_train.shape[0], X_val.shape[0], X_test.shape[0]]
        })
        st.table(summary_df)
        st.success("✅ Dữ liệu đã được chia thành công!")
        st.dataframe(X_train.head())

        return X_train, X_val, X_test, y_train, y_val, y_test

    return None, None, None, None, None, None

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

    # Lọc tất cả các cột số
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

    if not numerical_cols:
        st.success("✅ Không có thuộc tính dạng số cần chuẩn hóa!")
        return df

    # Chuẩn hóa tất cả các cột số
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Lưu lại trong session_state để tránh mất dữ liệu khi tải lại trang
    st.session_state.df = df

    st.success(f"✅ Đã chuẩn hóa tất cả các cột số: {', '.join(numerical_cols)}")
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
        - **Cabin**: Cột này có quá nhiều giá trị bị thiếu 687/891 .
        - **Ticket**: Mã vé không mang nhiều thông tin hữu ích và có 681/891 vé khác nhau.
        - **Name**:  Không cần thiết cho bài toán dự đoán sống sót.
        ```python
            columns_to_drop = ["Cabin", "Ticket", "Name"]  
            df.drop(columns=columns_to_drop, inplace=True)
        ```
        """)
    df1=drop(df)
    
    st.subheader("2️⃣ Xử lý giá trị thiếu")
    st.write("""
        Dữ liệu thực tế thường có giá trị bị thiếu. Ta cần xử lý như điền vào nan bằng trung bình hoặc trung vị có thể xóa nếu số dòng dữ liệu thiếu ít ,để tránh ảnh hưởng đến mô hình.
        - **Cột "Age"**: Có thể điền trung bình hoặc trung vị .
        - **Cột "Fare"**: Có thể điền giá trị trung bình hoặc trung vị .
        - **Cột "Embarked"**:   Xóa các dòng bị thiếu vì số lượng ít 2/891.
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
        - **Cột "Sex"**: Chuyển thành 1 (male), 0 (female).
        - **Cột "Embarked"**:   Chuyển thành 1 (Q), 2 (S), 3 (C).
        ```python
            df["Sex"] = df["Sex"].map({"male": 1, "female": 0})  # Mã hóa giới tính
            df['Embarked'] = df['Embarked'].map({'Q': 0, 'S': 1, 'C': 2})

        ```
        """)

    df=chuyen_doi_kieu_du_lieu(df)

    st.subheader("4️⃣ Chuẩn hóa dữ liệu số")
    st.write("""
        Các giá trị số có thể có khoảng giá trị khác nhau, làm ảnh hưởng đến mô hình. Ta sẽ chuẩn hóa toàn bộ về cùng một thang đo bằng StandardScaler.
        
        ```python
            scaler = StandardScaler()
            df[["Age", "Fare",...]] = scaler.fit_transform(df[["Age", "Fare",...]])

        ```
        """)

    
    df=chuan_hoa_du_lieu(df)
    
    st.subheader("5️⃣ Chia dữ liệu thành tập Train, Validation, và Test")
    st.write("""
    ### 📌 Chia tập dữ liệu
    Dữ liệu được chia thành ba phần để đảm bảo mô hình tổng quát tốt:
    - **70%**: để train mô hình.
    - **15%**: để validation, dùng để điều chỉnh tham số.
    - **15%**: để test, đánh giá hiệu suất thực tế.

    ```python
    from sklearn.model_selection import train_test_split

    # Chia dữ liệu theo tỷ lệ 85% (Train) - 15% (Test)
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

    # Chia tiếp 15% của Train để làm Validation (~12.75% của toàn bộ dữ liệu)
    val_size = 0.15 / 0.85  
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=val_size, stratify=y_train_full, random_state=42)
    ```
    """)
       
    X_train, X_val, X_test, y_train, y_val, y_test =train_test_size(df)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_multiple_linear_regression(X_train, y_train, learning_rate=0.001, n_iterations=200):
    """Huấn luyện hồi quy tuyến tính bội bằng Gradient Descent."""
    
    # Chuyển đổi X_train, y_train sang NumPy array để tránh lỗi
    X_train = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
    y_train = y_train.to_numpy().reshape(-1, 1) if isinstance(y_train, (pd.Series, pd.DataFrame)) else y_train.reshape(-1, 1)

    # Kiểm tra NaN hoặc Inf
    if np.isnan(X_train).any() or np.isnan(y_train).any():
        raise ValueError("Dữ liệu đầu vào chứa giá trị NaN!")
    if np.isinf(X_train).any() or np.isinf(y_train).any():
        raise ValueError("Dữ liệu đầu vào chứa giá trị vô cùng (Inf)!")

    # Chuẩn hóa dữ liệu để tránh tràn số
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Lấy số lượng mẫu (m) và số lượng đặc trưng (n)
    m, n = X_train.shape
    #st.write(f"Số lượng mẫu (m): {m}, Số lượng đặc trưng (n): {n}")

    # Thêm cột bias (x0 = 1) vào X_train
    X_b = np.c_[np.ones((m, 1)), X_train]
    #st.write(f"Kích thước ma trận X_b: {X_b.shape}")

    # Khởi tạo trọng số ngẫu nhiên nhỏ
    w = np.random.randn(X_b.shape[1], 1) * 0.01  
    #st.write(f"Trọng số ban đầu: {w.flatten()}")

    # Gradient Descent
    for iteration in range(n_iterations):
        gradients = (2/m) * X_b.T.dot(X_b.dot(w) - y_train)

        # Kiểm tra xem gradients có NaN không
        # st.write(gradients)
        if np.isnan(gradients).any():
            raise ValueError("Gradient chứa giá trị NaN! Hãy kiểm tra lại dữ liệu hoặc learning rate.")

        w -= learning_rate * gradients

    #st.success("✅ Huấn luyện hoàn tất!")
    #st.write(f"Trọng số cuối cùng: {w.flatten()}")
    return w
def train_polynomial_regression(X_train, y_train, degree=2, learning_rate=0.001, n_iterations=500):
    """Huấn luyện hồi quy đa thức **không có tương tác** bằng Gradient Descent."""

    # Chuyển dữ liệu sang NumPy array nếu là pandas DataFrame/Series
    X_train = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
    y_train = y_train.to_numpy().reshape(-1, 1) if isinstance(y_train, (pd.Series, pd.DataFrame)) else y_train.reshape(-1, 1)

    # Tạo đặc trưng đa thức **chỉ thêm bậc cao, không có tương tác**
    X_poly = np.hstack([X_train] + [X_train**d for d in range(2, degree + 1)])
    # Chuẩn hóa dữ liệu để tránh tràn số
    scaler = StandardScaler()
    X_poly = scaler.fit_transform(X_poly)

    # Lấy số lượng mẫu (m) và số lượng đặc trưng (n)
    m, n = X_poly.shape
    print(f"Số lượng mẫu (m): {m}, Số lượng đặc trưng (n): {n}")

    # Thêm cột bias (x0 = 1)
    X_b = np.c_[np.ones((m, 1)), X_poly]
    print(f"Kích thước ma trận X_b: {X_b.shape}")

    # Khởi tạo trọng số ngẫu nhiên nhỏ
    w = np.random.randn(X_b.shape[1], 1) * 0.01  
    print(f"Trọng số ban đầu: {w.flatten()}")

    # Gradient Descent
    for iteration in range(n_iterations):
        gradients = (2/m) * X_b.T.dot(X_b.dot(w) - y_train)

        # Kiểm tra nếu gradient có giá trị NaN
        if np.isnan(gradients).any():
            raise ValueError("Gradient chứa giá trị NaN! Hãy kiểm tra lại dữ liệu hoặc learning rate.")

        w -= learning_rate * gradients

    print("✅ Huấn luyện hoàn tất!")
    print(f"Trọng số cuối cùng: {w.flatten()}")
    
    return w

def chon_mo_hinh(model_type, X_train, X_test, y_train, y_test, n_folds=5):
    """Chọn mô hình hồi quy tuyến tính bội hoặc hồi quy đa thức."""
    degree = 2
    fold_mse = []  # Danh sách MSE của từng fold
    scaler = StandardScaler()  # Chuẩn hóa dữ liệu cho hồi quy đa thức nếu cần
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train, y_train)):
        X_train_fold, X_valid = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_train_fold, y_valid = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        st.write("🚀 Fold {fold + 1}: Train size = {len(X_train_fold)}, Validation size = {len(X_valid)}")

        if model_type == "linear":
            w= train_multiple_linear_regression(X_train_fold, y_train_fold)

            w = np.array(w).reshape(-1, 1)
            
            X_valid = X_valid.to_numpy()


            X_valid_b = np.c_[np.ones((len(X_valid), 1)), X_valid]  # Thêm bias
            y_valid_pred = X_valid_b.dot(w)  # Dự đoán
        elif model_type == "polynomial":
            
            X_train_fold = scaler.fit_transform(X_train_fold)
                
            w = train_polynomial_regression(X_train_fold, y_train_fold, degree)
            
            w = np.array(w).reshape(-1, 1)
            
            X_valid_scaled = scaler.transform(X_valid.to_numpy())
            X_valid_poly = np.hstack([X_valid_scaled] + [X_valid_scaled**d for d in range(2, degree + 1)])
            X_valid_b = np.c_[np.ones((len(X_valid_poly), 1)), X_valid_poly]
            
            y_valid_pred = X_valid_b.dot(w)  # Dự đoán
        else:
            raise ValueError("⚠️ Chọn 'linear' hoặc 'polynomial'!")

        mse = mean_squared_error(y_valid, y_valid_pred)
        fold_mse.append(mse)

        print(f"📌 Fold {fold + 1} - MSE: {mse:.4f}")

    # 🔥 Huấn luyện lại trên toàn bộ tập train
    if model_type == "linear":
        final_w = train_multiple_linear_regression(X_train, y_train)
        X_test_b = np.c_[np.ones((len(X_test), 1)), X_test]
        y_test_pred = X_test_b.dot(final_w)
    else:
        X_train_scaled = scaler.fit_transform(X_train)
        final_w = train_polynomial_regression(X_train_scaled, y_train, degree)

        X_test_scaled = scaler.transform(X_test.to_numpy())
        X_test_poly = np.hstack([X_test_scaled] + [X_test_scaled**d for d in range(2, degree + 1)])
        X_test_b = np.c_[np.ones((len(X_test_poly), 1)), X_test_poly]

        y_test_pred = X_test_b.dot(final_w)

    # 📌 Đánh giá trên tập test
    test_mse = mean_squared_error(y_test, y_test_pred)
    avg_mse = np.mean(fold_mse)  # Trung bình MSE qua các folds

    st.success(f"MSE trung bình qua các folds: {avg_mse:.4f}")
    st.success(f"MSE trên tập test: {test_mse:.4f}")

    return final_w, avg_mse, scaler




def main():
    uploaded_file = st.file_uploader("📂 Chọn file dữ liệu (.csv hoặc .txt)", type=["csv", "txt"])
    if uploaded_file is not None:  # Kiểm tra xem file đã được tải lên chư
        try:
            df = pd.read_csv(uploaded_file, delimiter=",")
            
            X_train, X_val, X_test, y_train, y_val, y_test=hien_thi_ly_thuyet(df)
            
            
            model_type = st.radio("Chọn loại mô hình:", ["Multiple Linear Regression", "Polynomial Regression"])

            # Khi nhấn nút sẽ huấn luyện mô hình
            if st.button("Huấn luyện mô hình"):
            
                model_type_value = "linear" if model_type == "Multiple Linear Regression" else "polynomial"

                # Gọi hàm với đúng thứ tự tham số
                final_w, avg_mse, scaler = chon_mo_hinh(model_type_value, X_train, X_test, y_train, y_test)
            
            
            
            
        except Exception as e:
            st.error(f"❌ Lỗi khi đọc file: {e}")
    
        


        


            
  

if __name__ == "__main__":
    main()
