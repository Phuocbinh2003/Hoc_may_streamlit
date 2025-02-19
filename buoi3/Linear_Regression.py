import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Tiêu đề
def bt_buoi3():
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