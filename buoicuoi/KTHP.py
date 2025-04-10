import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import mlflow

exp = mlflow.set_experiment("alphabet_data_processing_experiment")
with mlflow.start_run(experiment_id=exp.experiment_id):

    def tien_xu_ly_du_lieu(updates_file=None):
        if updates_file is not None:
            df = pd.read_csv(updates_file)
        else:
            # Giả sử bạn đang làm việc với các file npy (alphabet_X.npy và alphabet_y.npy)
            
            X = np.load('buoicuoi/alphabet_X.npy', allow_pickle=True)
            y = np.load('buoicuoi/alphabet_X.npy', allow_pickle=True)
            df = pd.DataFrame(X, columns=["Feature_" + str(i) for i in range(X.shape[1])])
            df['Target'] = y

        # Xử lý các cột kiểu chữ (alphabet) bằng LabelEncoder
        label_encoder = LabelEncoder()
        df['Target'] = label_encoder.fit_transform(df['Target'])  # Chuyển đổi cột 'Target' sang dạng số

        # Hoặc bạn có thể dùng OneHotEncoder nếu cần
        # one_hot_encoder = OneHotEncoder(sparse=False)
        # df['Target'] = one_hot_encoder.fit_transform(df[['Target']])

        # Tiền xử lý các cột khác nếu có dữ liệu kiểu alphabet
        for column in df.select_dtypes(include=['object']).columns:
            df[column] = label_encoder.fit_transform(df[column])

        # Chuẩn hóa dữ liệu số
        scaler = StandardScaler()
        df[df.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(df.select_dtypes(include=[np.number]))

        return df

    def test_train_size(actual_train_ratio, val_ratio_within_train, test_ratio):
        df = tien_xu_ly_du_lieu()
        X = df.drop(columns=['Target'])
        y = df['Target']
   
        # Chuyển đổi tỷ lệ phần trăm thành giá trị thực
        actual_train_size = actual_train_ratio / 100
        test_size = test_ratio / 100
        val_size = (val_ratio_within_train / 100) * actual_train_size  # Validation từ tập Train
        
        # Chia tập Train-Test trước
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

        # Tiếp tục chia tập Train thành Train-Validation
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size / actual_train_size, stratify=y_train, random_state=42)
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def chon_mo_hinh(model_type, X_train, X_val, X_test, y_train, y_val, y_test):
        """Chọn mô hình hồi quy tuyến tính bội hoặc hồi quy đa thức."""
        from sklearn.linear_model import LinearRegression
        fold_mse = []

        # Đảm bảo rằng mô hình hoạt động đúng với dữ liệu đã xử lý
        if model_type == "linear":
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_val_pred = model.predict(X_val)
        else:
            raise ValueError("⚠️ Chọn 'linear' hoặc 'polynomial'!")

        mse = mean_squared_error(y_val, y_val_pred)
        fold_mse.append(mse)

        st.success(f"MSE trung bình qua các folds: {np.mean(fold_mse):.4f}")
        st.write(f"MSE trên tập test: {mean_squared_error(y_test, y_val_pred):.4f}")

        return model

    def kthp():
        st.title("🔍 Tiền xử lý dữ liệu - Alphabet")

        uploaded_file = st.file_uploader("📂 Chọn file dữ liệu (.csv hoặc .txt)", type=["csv", "txt"])
        if uploaded_file is not None:
            df = tien_xu_ly_du_lieu(uploaded_file)
            st.write(df.head(10))

        train_ratio = st.slider("Chọn tỷ lệ Train (%)", min_value=50, max_value=90, value=70, step=1)
        test_ratio = 100 - train_ratio  # Test tự động tính toán

        val_ratio_within_train = st.slider("Chọn tỷ lệ Validation trong Train (%)", min_value=0, max_value=50, value=30, step=1)

        # Tính toán lại tỷ lệ Validation trên toàn bộ dataset
        val_ratio = (val_ratio_within_train / 100) * train_ratio
        actual_train_ratio = train_ratio - val_ratio

        # Hiển thị kết quả
        st.write(f"Tỷ lệ dữ liệu: Train = {actual_train_ratio:.1f}%, Validation = {val_ratio:.1f}%, Test = {test_ratio:.1f}%")

        X_train, X_val, X_test, y_train, y_val, y_test = test_train_size(actual_train_ratio, val_ratio_within_train, test_ratio)

        # Chọn mô hình    
        model_type = st.radio("Chọn loại mô hình:", ["Multiple Linear Regression", "Polynomial Regression"])

        # Khi nhấn nút sẽ huấn luyện mô hình
        if st.button("Huấn luyện mô hình"):
            model_type_value = "linear" if model_type == "Multiple Linear Regression" else "polynomial"
            final_model = chon_mo_hinh(model_type_value, X_train, X_val, X_test, y_train, y_val, y_test)

if __name__ == "__main__":
    kthp()
