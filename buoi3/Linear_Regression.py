import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import zscore
# Tiêu đề
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

    st.latex(r"""
    y = w_0 + w_1x_1 + w_2x_2 + \dots + w_nx_n
    """)

    st.write("Multiple Linear Regression là mô hình tuyến tính sử dụng nhiều biến độc lập để dự đoán biến phụ thuộc.")

    # Giới thiệu Polynomial Regression
    st.write("## 2. Polynomial Regression")

    st.latex(r"""
    y = w_0 + w_1x + w_2x^2 + w_3x^3 + \dots + w_nx^n
    """)

    st.write("Polynomial Regression mở rộng mô hình tuyến tính bằng cách thêm các bậc cao hơn của biến đầu vào.")

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
if __name__ == "__main__":
    bt_buoi3()