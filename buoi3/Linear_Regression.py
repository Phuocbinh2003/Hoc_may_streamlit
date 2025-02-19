import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# Tiêu đề


def tien_xu_ly_du_lieu():
    df = pd.read_csv("buoi2/data.txt")

    # Loại bỏ các cột không cần thiết
    columns_to_drop = ["Cabin", "Ticket", "Name"]  # Cột không cần thiết
    df.drop(columns=columns_to_drop, inplace=True)  # Loại bỏ cột
    # Xử lý giá trị thiếu
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df.dropna(subset=['Embarked'], inplace=True)  # Xóa dòng nếu 'Embarked' bị thiếu

    # Mã hóa giới tính: Male -> 1, Female -> 0
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

    # Mã hóa 'Embarked' bằng One-Hot Encoding
    df['Embarked'] = df['Embarked'].map({'Q': 0, 'S': 1, 'C': 2})
    

    # Chuẩn hóa các giá trị số
    scaler = StandardScaler()
    df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

    # Chia dữ liệu thành đầu vào (X) và nhãn (y)
    X = df.drop(columns=['Survived'])
    y = df['Survived']

    # Chia tập train (70%), validation (15%), test (15%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

    # 2️⃣ Dùng StratifiedKFold với mỗi fold chọn 15% làm validation
    kf = StratifiedKFold(n_splits=int(1 / 0.15), shuffle=True, random_state=42)
    return X_train, X_test, y_train, y_test, kf ,df


def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    m, n = X.shape
    X_b = np.c_[np.ones((m, 1)), X]  # Thêm cột bias
    w = np.random.randn(n + 1, 1)  # Khởi tạo trọng số ngẫu nhiên

    for iteration in range(n_iterations):
        gradients = 2/m * X_b.T.dot(X_b.dot(w) - y)
        w -= learning_rate * gradients
    
    return w

def train_multiple_linear_regression(X_train, y_train, learning_rate=0.01, n_iterations=1000):
    """Huấn luyện hồi quy tuyến tính bội bằng Gradient Descent."""
    return gradient_descent(X_train, y_train, learning_rate, n_iterations)

def train_polynomial_regression(X_train, y_train, degree=2, learning_rate=0.01, n_iterations=1000):
    """Huấn luyện hồi quy đa thức bằng Gradient Descent."""
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    w = gradient_descent(X_train_poly, y_train, learning_rate, n_iterations)
    return w, poly  # Trả về cả trọng số và đối tượng poly để transform tập test

def chon_mo_hinh(model_type="linear", degree=2, learning_rate=0.01, n_iterations=1000):
    """Chọn mô hình hồi quy tuyến tính bội hoặc hồi quy đa thức."""
    X_train_full, X_test, y_train_full, y_test, kf, df = tien_xu_ly_du_lieu()
    fold_mse = []
    poly = None

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train_full, y_train_full)):
        X_train, X_valid = X_train_full.iloc[train_idx], X_train_full.iloc[valid_idx]
        y_train, y_valid = y_train_full.iloc[train_idx], y_train_full.iloc[valid_idx]

        print(f"\n🚀 Fold {fold + 1}: Train size = {len(X_train)}, Validation size = {len(X_valid)}")

        if model_type == "linear":
            w = train_multiple_linear_regression(X_train, y_train, learning_rate, n_iterations)
            X_valid_b = np.c_[np.ones((len(X_valid), 1)), X_valid]
            y_valid_pred = X_valid_b.dot(w).flatten()
        elif model_type == "polynomial":
            w, poly = train_polynomial_regression(X_train, y_train, degree, learning_rate, n_iterations)
            X_valid_poly = poly.transform(X_valid)
            X_valid_poly_b = np.c_[np.ones((len(X_valid_poly), 1)), X_valid_poly]
            y_valid_pred = X_valid_poly_b.dot(w).flatten()
        else:
            raise ValueError("⚠️ Chọn 'linear' hoặc 'polynomial'!")
        
        mse = mean_squared_error(y_valid, y_valid_pred)
        fold_mse.append(mse)
        print(f"📌 Fold {fold + 1} - MSE: {mse:.4f}")
    
    # 🔥 Huấn luyện lại trên toàn bộ tập train_full
    if model_type == "linear":
        final_w = train_multiple_linear_regression(X_train_full, y_train_full, learning_rate, n_iterations)
        X_test_b = np.c_[np.ones((len(X_test), 1)), X_test]
        y_test_pred = X_test_b.dot(final_w).flatten()
    else:
        final_w, poly = train_polynomial_regression(X_train_full, y_train_full, degree, learning_rate, n_iterations)
        X_test_poly = poly.transform(X_test)
        X_test_poly_b = np.c_[np.ones((len(X_test_poly), 1)), X_test_poly]
        y_test_pred = X_test_poly_b.dot(final_w).flatten()
    
    test_mse = mean_squared_error(y_test, y_test_pred)
    avg_mse = np.mean(fold_mse)
    
    st.success(f"MSE trung bình qua các folds: {avg_mse:.4f}")
    st.success(f"MSE trên tập test: {test_mse:.4f}")
    
    return final_w, avg_mse, poly

def bt_buoi3():
    uploaded_file = "buoi2/data.txt"
    try:
        df = pd.read_csv(uploaded_file, delimiter=",")
    except FileNotFoundError:
        st.error("❌ Không tìm thấy tệp dữ liệu. Vui lòng kiểm tra lại đường dẫn.")
        st.stop()
    st.title("🔍 Tiền xử lý dữ liệu")
    
    st.subheader("📌 10 dòng đầu của dữ liệu gốc")
    st.write(df.head(10))
    
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

                # Hiển thị báo cáo lỗ
    st.table(error_report)

                # Hiển thị số lượng dữ liệu trùng lặp
    st.write(f"🔁 **Số lượng dòng bị trùng lặp:** {duplicate_count}")         
    
    st.title("🔍 Tiền xử lý dữ liệu")

    # Loại bỏ các cột không cần thiết
    st.subheader("1️⃣ Loại bỏ các cột không quan trọng")
    st.write("""
    Một số cột trong dữ liệu có thể không đóng góp nhiều vào kết quả dự đoán hoặc chứa quá nhiều giá trị thiếu. Việc loại bỏ các cột này giúp giảm độ phức tạp của mô hình và cải thiện hiệu suất.
    """)

    # Xử lý giá trị thiếu
    st.subheader("2️⃣ Xử lý giá trị thiếu")
    st.write("""
    Dữ liệu thực tế thường chứa các giá trị bị thiếu. Ta cần lựa chọn phương pháp thích hợp như điền giá trị trung bình, loại bỏ hàng hoặc sử dụng mô hình dự đoán để xử lý chúng nhằm tránh ảnh hưởng đến mô hình.
    """)

    # Chuyển đổi kiểu dữ liệu
    st.subheader("3️⃣ Chuyển đổi kiểu dữ liệu")
    st.write("""
    Một số cột trong dữ liệu có thể chứa giá trị dạng chữ (danh mục). Để mô hình có thể xử lý, ta cần chuyển đổi chúng thành dạng số bằng các phương pháp như one-hot encoding hoặc label encoding.
    """)

    # Chuẩn hóa dữ liệu số
    st.subheader("4️⃣ Chuẩn hóa dữ liệu số")
    st.write("""
    Các giá trị số trong tập dữ liệu có thể có phạm vi rất khác nhau, điều này có thể ảnh hưởng đến độ hội tụ của mô hình. Ta cần chuẩn hóa dữ liệu để đảm bảo tất cả các đặc trưng có cùng trọng số khi huấn luyện mô hình.
    """)

    # Chia dữ liệu thành tập Train, Validation, và Test
    st.subheader("5️⃣ Chia dữ liệu thành tập Train, Validation, và Test")
    st.write("""
    Để đảm bảo mô hình hoạt động tốt trên dữ liệu thực tế, ta chia tập dữ liệu thành ba phần:
    - **Train**: Dùng để huấn luyện mô hình.
    - **Validation**: Dùng để điều chỉnh tham số mô hình nhằm tối ưu hóa hiệu suất.
    - **Test**: Dùng để đánh giá hiệu suất cuối cùng của mô hình trên dữ liệu chưa từng thấy.
    """)
    
    
    
    st.title("Lựa chọn thuật toán học máy: Multiple vs. Polynomial Regression")

    # Giới thiệu
    st.write("## 1. Multiple Linear Regression")
    st.write("""
    Hồi quy tuyến tính bội là một thuật toán học máy có giám sát, mô tả mối quan hệ giữa một biến phụ thuộc (output) và nhiều biến độc lập (input) thông qua một hàm tuyến tính.
    Ví dụ dự đoán giá nhà dựa trên diện tích, số phòng, vị trí, ... 
    
    Công thức tổng quát của mô hình hồi quy tuyến tính bội:
    """)
    st.image("buoi3/img1.png", caption="Multiple Linear Regression đơn", use_container_width =True)
    st.latex(r"""
    y = w_0 + w_1x_1 + w_2x_2 + \dots + w_nx_n
    """)

    
   
   

    # Giới thiệu Polynomial Regression
    st.write("## 2. Polynomial Regression")

    st.write("Polynomial Regression mở rộng mô hình tuyến tính bằng cách thêm các bậc cao hơn của biến đầu vào.")
    
    st.image("buoi3/img3.png", caption="Polynomial Regression ", use_container_width =True)
    st.write("""
     Công thức tổng quát của mô hình hồi quy tuyến tính bội:
    """)
    st.latex(r"""
    y = w_0 + w_1x + w_2x^2 + w_3x^3 + \dots + w_nx^n
    """)

    
    st.write("""
    ### Hàm mất mát (Loss Function) của Linear Regression
    Hàm mất mát phổ biến nhất là **Mean Squared Error (MSE)**:
    """)
    st.latex(r"""
    MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
    """)

    st.markdown(r"""
    Trong đó:
    - $n$: Số lượng điểm dữ liệu.
    - $y_i$: Giá trị thực tế của biến phụ thuộc.
    - $\hat{y}_i$: Giá trị dự đoán từ mô hình.
    """)

    st.markdown(r"""
    Mục tiêu của hồi quy tuyến tính bội là tìm các hệ số trọng số $w_0, w_1, w_2, ..., w_n$ sao cho giá trị MSE nhỏ nhất.

    ### Thuật toán Gradient Descent
    1. Khởi tạo các trọng số $w_0, w_1, w_2, ..., w_n$ với giá trị bất kỳ.
    2. Tính gradient của MSE đối với từng trọng số.
    3. Cập nhật trọng số theo quy tắc của thuật toán Gradient Descent.

    ### Đánh giá mô hình hồi quy tuyến tính bội
    - **Hệ số tương quan (R)**: Đánh giá mức độ tương quan giữa giá trị thực tế và giá trị dự đoán.
    - **Hệ số xác định (R²)**: Đo lường phần trăm biến động của biến phụ thuộc có thể giải thích bởi các biến độc lập:
    """)
    st.latex(r"""
    R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
    """)

    st.write("""
    - **Adjusted R²**: Điều chỉnh cho số lượng biến độc lập, giúp tránh overfitting:
    """)
    st.latex(r"""
    R^2_{adj} = 1 - \left( \frac{(1 - R^2)(n - 1)}{n - k - 1} \right)
    """)

    st.markdown(r"""
    Trong đó:
    - $n$: Số lượng quan sát.
    - $k$: Số lượng biến độc lập.
    - $\bar{y}$: Giá trị trung bình của biến phụ thuộc.
    """)

    st.write("""
    
    - **Sai số chuẩn (SE)**: Đánh giá mức độ phân tán của sai số dự đoán quanh giá trị thực tế:
    """)
    st.latex(r"""
    SE = \sqrt{\frac{\sum (y_i - \hat{y}_i)^2}{n - k - 1}}
    """)

    st.write("""
    Các chỉ số này giúp đánh giá độ chính xác và khả năng khái quát hóa của mô hình hồi quy tuyến tính bội.
    """)
    # Vẽ biểu đồ so sánh
    st.write("## 3. Minh họa trực quan")

    # Tạo dữ liệu mẫu
    np.random.seed(0)
    x = np.sort(5 * np.random.rand(20, 1), axis=0)
    y = 2 * x**2 - 3 * x + np.random.randn(20, 1) * 2

    # Hồi quy tuyến tính
    lin_reg = LinearRegression()
    lin_reg.fit(x, y)
    y_pred_linear = lin_reg.predict(x)

    # Hồi quy bậc hai
    poly_features = PolynomialFeatures(degree=2)
    x_poly = poly_features.fit_transform(x)
    poly_reg = LinearRegression()
    poly_reg.fit(x_poly, y)
    y_pred_poly = poly_reg.predict(x_poly)

    # Vẽ biểu đồ
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, color='blue', label='Dữ liệu thực tế')
    ax.plot(x, y_pred_linear, color='red', label='Multiple Linear Regression')
    ax.plot(x, y_pred_poly, color='green', label='Polynomial Regression (bậc 2)')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    st.pyplot(fig)
    
    X_train_full, X_test, y_train_full, y_test, kf, df = tien_xu_ly_du_lieu()
    st.write(df.head(10))

    # Chọn loại mô hình
    model_type = st.radio("Chọn loại mô hình:", ["Multiple Linear Regression", "Polynomial Regression"])

    # Nếu chọn Polynomial Regression, cho phép chọn bậc đa thức
    degree = 2
    if model_type == "Polynomial Regression":
        degree = st.slider("Chọn bậc của hồi quy đa thức:", min_value=2, max_value=5, value=2)

    # Khi nhấn nút sẽ huấn luyện mô hình
    if st.button("Huấn luyện mô hình"):
        model, avg_mse, poly = chon_mo_hinh(
            model_type="linear" if model_type == "Multiple Linear Regression" else "polynomial",
            degree=degree
        )

        # Hiển thị kết quả huấn luyện
        st.success(f"MSE trung bình qua các folds: {avg_mse:.4f}")

        # Nếu là Polynomial Regression, hiển thị thêm bậc của mô hình
        if model_type == "Polynomial Regression":
            st.write(f"✅ Mô hình hồi quy bậc {degree} đã được huấn luyện thành công!")
    
    
if __name__ == "__main__":
    bt_buoi3()