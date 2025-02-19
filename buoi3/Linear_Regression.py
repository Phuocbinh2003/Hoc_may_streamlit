import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

def tien_xu_ly_du_lieu():
    # Đọc dữ liệu
    try:
        df = pd.read_csv("buoi2/data.txt")
    except FileNotFoundError:
        st.error("❌ Không tìm thấy tệp dữ liệu. Vui lòng kiểm tra lại đường dẫn.")
        st.stop()

    # Xử lý dữ liệu
    df = df.drop(columns=["Cabin", "Ticket", "Name"])
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df.dropna(subset=['Embarked'], inplace=True)
    
    # Mã hóa dữ liệu
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
    df['Embarked'] = df['Embarked'].map({'Q': 0, 'S': 1, 'C': 2})
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
    
    # Tách features và target
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    
    # Chia tập dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, df

def initialize_weights(n_features):
    return np.random.randn(n_features, 1)

def gradient_descent(X, y, w, learning_rate, n_iterations):
    m = len(y)
    for _ in range(n_iterations):
        y_pred = X.dot(w)
        error = y_pred - y
        gradients = (2/m) * X.T.dot(error)
        w -= learning_rate * gradients
    return w

def train_linear_regression(X_train, y_train, learning_rate=0.001, n_iter=200):
    # Thêm cột bias
    X_b = np.c_[np.ones((len(X_train), 1)), X_train]
    
    # Khởi tạo trọng số
    w = initialize_weights(X_b.shape[1])
    
    # Huấn luyện
    y_train = y_train.values.reshape(-1, 1)
    return gradient_descent(X_b, y_train, w, learning_rate, n_iter)

def train_poly_regression(X_train, y_train, degree=2, learning_rate=0.001, n_iter=200):
    # Tạo đặc trưng đa thức
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly.fit_transform(X_train)
    
    # Khởi tạo trọng số
    w = initialize_weights(X_poly.shape[1])
    
    # Huấn luyện
    y_train = y_train.values.reshape(-1, 1)
    return gradient_descent(X_poly, y_train, w, learning_rate, n_iter), poly

def evaluate_model(model_type, X_train, y_train, X_test, y_test, learning_rate):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        # Chuẩn bị dữ liệu
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]
        
        # Huấn luyện mô hình
        if model_type == "Linear":
            w = train_linear_regression(X_tr, y_tr, learning_rate)
            X_val_b = np.c_[np.ones((len(X_val), 1)), X_val]
            y_pred = X_val_b.dot(w)
        else:
            w, poly = train_poly_regression(X_tr, y_tr, learning_rate=learning_rate)
            X_val_poly = poly.transform(X_val)
            y_pred = X_val_poly.dot(w)
        
        # Tính MSE
        fold_scores.append(mean_squared_error(y_val, y_pred))
    
    # Đánh giá trên tập test
    if model_type == "Linear":
        w_final = train_linear_regression(X_train, y_train, learning_rate)
        X_test_b = np.c_[np.ones((len(X_test), 1)), X_test]
        y_test_pred = X_test_b.dot(w_final)
    else:
        w_final, poly = train_poly_regression(X_train, y_train, learning_rate=learning_rate)
        X_test_poly = poly.transform(X_test)
        y_test_pred = X_test_poly.dot(w_final)
    
    test_mse = mean_squared_error(y_test, y_test_pred)
    return np.mean(fold_scores), test_mse

def bt_buoi3():
    st.title("🏆 So sánh Multiple và Polynomial Regression")
    
    # Tiền xử lý dữ liệu
    X_train, X_test, y_train, y_test, df = tien_xu_ly_du_lieu()
    
    # Hiển thị dữ liệu
    st.subheader("📊 Dữ liệu đã xử lý")
    st.dataframe(df.head())
    
    # Giao diện người dùng
    model_type = st.radio("Chọn loại mô hình:", ["Linear", "Polynomial"])
    learning_rate = st.slider("Tốc độ học", 0.0001, 0.01, 0.001, step=0.0001)
    
    if st.button("🏃♂️ Huấn luyện"):
        with st.spinner("Đang huấn luyện..."):
            avg_val_mse, test_mse = evaluate_model(
                model_type, 
                X_train, 
                y_train,
                X_test,
                y_test,
                learning_rate
            )
            
        st.success(f"📊 MSE Validation trung bình: {avg_val_mse:.4f}")
        st.success(f"🧪 MSE Test: {test_mse:.4f}")
        
        # Visualization
        fig, ax = plt.subplots()
        ax.bar(["Validation", "Test"], [avg_val_mse, test_mse], color=['blue', 'orange'])
        ax.set_ylabel("MSE")
        ax.set_title("So sánh hiệu suất mô hình")
        st.pyplot(fig)

if __name__ == "__main__":
    bt_buoi3()