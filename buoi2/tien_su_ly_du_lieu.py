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

    X, y = df.drop(columns=[selected_label]), df[selected_label]  # Mặc định
    
    if st.button("✅ Xác nhận Label"):
        st.session_state.target_column = selected_label
        X, y = df.drop(columns=[selected_label]), df[selected_label]
        st.success(f"✅ Đã chọn cột: **{selected_label}**")
    
    return X, y

def train_test_size():
    
    
    if "df" not in st.session_state:
        st.error("❌ Dữ liệu chưa được tải lên!")
        st.stop()
    df = st.session_state.df  # Lấy dữ liệu từ session_state
    
    X, y = choose_label(df)
    st.subheader("📊 Chia dữ liệu Train - Validation - Test")   
    df = st.session_state.df
    test_size = st.slider("📌 Chọn % dữ liệu Test", 10, 50, 20)

    remaining_size = 100 - test_size
    val_size = st.slider("📌 Chọn % dữ liệu Validation (trong phần Train)", 0, 50, 15)

    st.write(f"📌 **Tỷ lệ phân chia:** Test={test_size}%, Validation={val_size}%, Train={remaining_size - val_size}%")
    if st.button("✅ Xác nhận Chia"):
        st.write(f"⏳ Đang chia dữ liệu...")  # Giúp debug xem có chạy vào đây không
        # Kiểm tra y có nhiều hơn 1 giá trị không trước khi stratify
        stratify_option = y if y.nunique() > 1 else None
        
        # Chia dữ liệu thành Test trước
        X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=test_size/100, stratify=stratify_option, random_state=42)
        st.write(X_train_full.shape ,y_train_full.shape)
        
        
        # Chia tiếp phần còn lại thành Train và Validation
        stratify_option = y_train_full if y_train_full.nunique() > 1 else None
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=val_size / (100 - test_size), stratify=stratify_option, random_state=42)
        st.write(f"📊 Kích thước tập Train: {X_train.shape[0]} mẫu")
        st.write(f"📊 Kích thước tập Validation: {X_val.shape[0]} mẫu")
        st.write(f"📊 Kích thước tập Test: {X_test.shape[0]} mẫu")
        # Lưu vào session_state
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.y = y

        # Hiển thị thông tin số lượng mẫu
        summary_df = pd.DataFrame({
            "Tập dữ liệu": ["Train", "Validation", "Test"],
            "Số lượng mẫu": [X_train.shape[0], X_val.shape[0], X_test.shape[0]]
        })
        st.table(summary_df)

        st.success("✅ Dữ liệu đã được chia thành công!")
        
    # if "X_train" in st.session_state:
    #     st.dataframe(st.session_state.X_train.head())


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

    # Lấy các cột dạng chuỗi (categorical columns)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Nếu không có cột dạng chuỗi, thông báo và trả lại DataFrame ban đầu
    if not categorical_cols:
        st.success("✅ Không có cột dạng chuỗi cần chuyển đổi!")
        return df

    # Cho phép người dùng chọn cột cần chuyển đổi
    selected_col = st.selectbox("📌 Chọn cột để chuyển đổi:", categorical_cols)
    unique_values = df[selected_col].unique()
    
    # Tạo từ điển lưu trữ giá trị thay thế cho mỗi giá trị độc nhất
    mapping_dict = {}
    
    # Nếu có ít hơn 5 giá trị độc nhất, yêu cầu người dùng nhập giá trị thay thế
    if len(unique_values) < 5:
        for val in unique_values:
            new_val = st.text_input(f"🔄 Nhập giá trị thay thế cho `{val}`:", key=f"{selected_col}_{val}")
            mapping_dict[val] = new_val

        # Khi người dùng nhấn nút "Chuyển đổi dữ liệu"
        if st.button("🚀 Chuyển đổi dữ liệu"):
            # Kiểm tra xem session_state đã có mảng lưu các mapping_dict chưa
            if "mapping_dicts" not in st.session_state:
                st.session_state.mapping_dicts = []  # Tạo một mảng rỗng nếu chưa có

            # Lưu thông tin cột (tên cột và mapping_dict) vào mảng
            column_info = {
                "column_name": selected_col,
                "mapping_dict": mapping_dict
            }
            st.session_state.mapping_dicts.append(column_info)

            # Chuyển đổi các giá trị trong cột
            df[selected_col] = df[selected_col].map(lambda x: mapping_dict.get(x, x))
            df[selected_col] = pd.to_numeric(df[selected_col], errors='coerce')  # Chuyển thành số
            
            # Lưu lại DataFrame đã chuyển đổi trong session_state
            st.session_state.df = df
            st.success(f"✅ Đã chuyển đổi cột `{selected_col}`")

    # Hiển thị DataFrame đã được chuyển đổi
    st.dataframe(df.head())
    
    # Hiển thị mảng các mapping_dict đã lưu trong session_state
    if "mapping_dicts" in st.session_state:
        st.write("Danh sách các cột đã chuyển đổi:")
        for col_info in st.session_state.mapping_dicts:
            st.write(f"Cột: `{col_info['column_name']}`")
            st.write(f"Mapping Dict: {col_info['mapping_dict']}")
    
    return df




def chuan_hoa_du_lieu(df):
    # st.subheader("📊 Chuẩn hóa dữ liệu với StandardScaler")

    # Lọc tất cả các cột số
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

    # Tìm các cột nhị phân (chỉ chứa 0 và 1)
    binary_cols = [col for col in numerical_cols if df[col].dropna().isin([0, 1]).all()]

    # Loại bỏ cột nhị phân khỏi danh sách cần chuẩn hóa
    cols_to_scale = list(set(numerical_cols) - set(binary_cols))

    if not cols_to_scale:
        st.success("✅ Không có thuộc tính dạng số cần chuẩn hóa!")
        return df

    if st.button("🚀 Thực hiện Chuẩn hóa"):
        scaler = StandardScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

        # Lưu vào session_state
        st.session_state.df = df

        st.success(f"✅ Đã chuẩn hóa các cột số (loại bỏ cột nhị phân): {', '.join(cols_to_scale)}")
        st.info(f"🚫 Giữ nguyên các cột nhị phân: {', '.join(binary_cols) if binary_cols else 'Không có'}")
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
    df=drop(df)
    
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
    df=xu_ly_gia_tri_thieu(df)

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
    
def chia():
    st.subheader("Chia dữ liệu thành tập Train, Validation, và Test")
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
       
    train_test_size()
    
    


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

def chon_mo_hinh():
    st.subheader("🔍 Chọn mô hình hồi quy")
    model_type_V = st.radio("Chọn loại mô hình:", ["Multiple Linear Regression", "Polynomial Regression"])
    
    # Xác định loại mô hình
    model_type = "linear" if model_type_V == "Multiple Linear Regression" else "polynomial"
    
    
    n_folds = st.slider("Chọn số folds (KFold Cross-Validation):", min_value=2, max_value=10, value=5)

    # Thanh trượt chọn tốc độ học (learning rate)
    learning_rate = st.slider("Chọn tốc độ học (learning rate):", min_value=0.0001, max_value=0.1, value=0.01, step=0.0001)
    
    ddegree = 2
    if model_type == "polynomial":
        degree = st.slider("Chọn bậc đa thức:", min_value=2, max_value=5, value=2)
    fold_mse = []
    scaler = StandardScaler()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Kiểm tra nếu dữ liệu có trong session_state
    if "X_train" not in st.session_state or st.session_state.X_train is None:
        st.warning("⚠️ Vui lòng chia dữ liệu trước khi huấn luyện mô hình!")
        return None, None, None

    # Lấy dữ liệu từ session_state
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test
    # st.write(X_train.shape ,y_train.shape)
    # st.dataframe(X_train)
    
    if st.button("Huấn luyện mô hình"):
        st.write("⏳ Đang huấn luyện mô hình...")  # Giúp debug xem có chạy vào đây không

        for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train, y_train)):
            X_train_fold, X_valid = X_train.iloc[train_idx], X_train.iloc[valid_idx]
            y_train_fold, y_valid = y_train.iloc[train_idx], y_train.iloc[valid_idx]

            if model_type == "linear":
                w = train_multiple_linear_regression(X_train_fold, y_train_fold, learning_rate=learning_rate)
                w = np.array(w).reshape(-1, 1)
                X_valid_b = np.c_[np.ones((len(X_valid), 1)), X_valid.to_numpy()]
                y_valid_pred = X_valid_b.dot(w)
            else:  # Polynomial Regression
                X_train_fold = scaler.fit_transform(X_train_fold)
                w = train_polynomial_regression(X_train_fold, y_train_fold, degree,learning_rate=learning_rate)
                w = np.array(w).reshape(-1, 1)

                X_valid_scaled = scaler.transform(X_valid.to_numpy())
                X_valid_poly = np.hstack([X_valid_scaled] + [X_valid_scaled**d for d in range(2, degree + 1)])
                X_valid_b = np.c_[np.ones((len(X_valid_poly), 1)), X_valid_poly]
                y_valid_pred = X_valid_b.dot(w)

            mse = mean_squared_error(y_valid, y_valid_pred)
            fold_mse.append(mse)
            print(f"📌 Fold {fold + 1} - MSE: {mse:.4f}")

        # Huấn luyện trên toàn bộ tập train
        
        if model_type == "linear":
            final_w = train_multiple_linear_regression(X_train, y_train,learning_rate=learning_rate)
            st.session_state['linear_model'] = final_w
            X_test_b = np.c_[np.ones((len(X_test), 1)), X_test]
            y_test_pred = X_test_b.dot(final_w)
            
            
            
        else:
            X_train_scaled = scaler.fit_transform(X_train)
            final_w = train_polynomial_regression(X_train_scaled, y_train, degree,learning_rate=learning_rate)
            st.session_state['polynomial_model'] = final_w
            
            X_test_scaled = scaler.transform(X_test.to_numpy())
            X_test_poly = np.hstack([X_test_scaled] + [X_test_scaled**d for d in range(2, degree + 1)])
            X_test_b = np.c_[np.ones((len(X_test_poly), 1)), X_test_poly]
            y_test_pred = X_test_b.dot(final_w)

        test_mse = mean_squared_error(y_test, y_test_pred)
        avg_mse = np.mean(fold_mse)

        st.success(f"MSE trung bình qua các folds: {avg_mse:.4f}")
        st.success(f"MSE trên tập test: {test_mse:.4f}")

        return final_w, avg_mse, scaler

    return None, None, None


import numpy as np
import streamlit as st

def test():
    # Kiểm tra xem mô hình đã được lưu trong session_state chưa
    model_type = st.selectbox("Chọn mô hình:", ["linear", "polynomial"])

    if model_type == "linear" and "linear_model" in st.session_state:
        model = st.session_state["linear_model"]
    elif model_type == "polynomial" and "polynomial_model" in st.session_state:
        model = st.session_state["polynomial_model"]
    else:
        st.warning("Mô hình chưa được huấn luyện.")
        return

    # Nhập các giá trị cho các cột của X_train
    X_train = st.session_state.X_train
    st.write(X_train.head())  # Đảm bảo bạn dùng session_state
    num_columns = len(X_train.columns)
    column_names = X_train.columns.tolist()

    st.write(f"Nhập các giá trị cho {num_columns} cột của X_train:")

    # Tạo các trường nhập liệu cho từng cột
    X_train_input = []
    binary_columns = [] 
    # Kiểm tra nếu có dữ liệu mapping_dicts trong session_state
    if "mapping_dicts" not in st.session_state:
        st.session_state.mapping_dicts = []

    # Duyệt qua các cột và kiểm tra nếu có thông tin chuyển đổi
    for i, column_name in enumerate(column_names):
        # Kiểm tra xem cột có nằm trong mapping_dicts không
        mapping_dict = None
        for column_info in st.session_state.mapping_dicts:
            if column_info["column_name"] == column_name:
                mapping_dict = column_info["mapping_dict"]
                break

        if mapping_dict:  # Nếu có mapping_dict, hiển thị dropdown với các giá trị thay thế
            value = st.selectbox(f"Giá trị cột {column_name}", options=list(mapping_dict.keys()), key=f"column_{i}")
            
            value = mapping_dict[value]  # Lấy giá trị thay thế tương ứng
            
        else:  # Nếu không có mapping_dict, yêu cầu người dùng nhập số
            value = st.number_input(f"Giá trị cột {column_name}", key=f"column_{i}")
            
        X_train_input.append(value)
    st.write(model)
    # Chuyển đổi list thành array
    X_train_input = np.array(X_train_input).reshape(1, -1)
    st.write(X_train_input)
    if np.all(np.isin(X_train_input, [0, 1])):
    # Nếu tất cả giá trị là 0 hoặc 1 thì không cần chuẩn hó
        X_train_input_final = X_train_input
    else:
        # Nếu có giá trị khác 0 và 1, chuẩn hó
        scaler = StandardScaler()
        # X_train_input_final = scaler.fit_transform(X_train_input.reshape(-1, 1)).flatten()
        X_train_input_final = scaler.fit_transform(X_train_input.T).T 

    # Hiển thị kết quả
    st.write("Dữ liệu sau khi xử lý:")
    st.write(X_train_input_final)
    
    # X_train_input_without_binary = np.delete(X_train_input, binary_columns, axis=1)
    # st.write(X_train_input_without_binary)
    # # Chuẩn hóa các cột không phải nhị phân
    # scaler = StandardScaler()
    # X_train_input_normalized = scaler.fit_transform(X_train_input_without_binary)
    # st.write(X_train_input_normalized)
    # # Kết hợp lại cột nhị phân và các cột đã chuẩn hóa
    # X_train_input_final = np.hstack([X_train_input_normalized, X_train_input[:, binary_columns]])
    # st.write(model)
    # st.write(X_train_input_final)
    # Dự đoán khi nhấn nút
    if st.button("Dự đoán"):
        # Thêm cột 1 cho intercept (nếu cần)
        X_input_b = np.c_[np.ones((X_train_input_final.shape[0], 1)), X_train_input_final]
        
        # Dự đoán với mô hình đã lưu
        st.write(X_input_b)
        
        y_pred = X_input_b.dot(model)  # Dự đoán với mô hình đã lưu
        
        # Hiển thị kết quả dự đoán
        st.write(f"Dự đoán: {y_pred[0]}")

    
def data():
    uploaded_file = st.file_uploader("📂 Chọn file dữ liệu (.csv hoặc .txt)", type=["csv", "txt"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, delimiter=",")
            st.success("📂 File tải lên thành công!")

            # Hiển thị lý thuyết và xử lý dữ liệu
            hien_thi_ly_thuyet(df)
        except Exception as e:
            st.error(f"❌ Lỗi : {e}")
def chon():
    try:
                
        final_w, avg_mse, scaler = chon_mo_hinh()
    except Exception as e:
        st.error(f"Lỗi xảy ra: {e}")
def main():
    
    tab1, tab2, tab3 = st.tabs(["📘 Tiền xử lý dữ liệu","⚙️ Huấn luyện", "🔢 Dự đoán"])
    with tab1:
        data()
    with tab2:
        chia()
        chon()
    with tab3:
        test()
    
    
            
            
            

        
if __name__ == "__main__":
    main()
    
        


        


            
  

