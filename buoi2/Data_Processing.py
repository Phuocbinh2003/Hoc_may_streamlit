import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Tải dữ liệu


def phan_gioi_thieu():
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
    
    return df, X_train, X_val, X_test, y_train, y_val, y_test
def phan_train(X_train, y_train, X_val, y_val, X_test, y_test):
    st.title("🚀 Huấn luyện mô hình")
    st.subheader(" mô hình Random Forest")
    st.write(f"""
        Mô hình Random Forest là một mô hình mạnh mẽ và linh hoạt, thường được sử dụng trong các bài toán phân loại và hồi quy.
        Ưu điểm:   
        - Xử lý tốt với dữ liệu lớn.
        - Không yêu cầu chuẩn hóa dữ liệu.
        - Dễ dàng xử lý overfitting.
        Nhược điểm:
        - Không hiệu quả với dữ liệu có nhiều giá trị thiếu.
        - Mất hiệu suất khi số lượng cây lớn.
        - Không thể hiển thị quá trình học.
        """)
    
    
    
    st.write("""
        Đến bước quan trọng nhất: huấn luyện mô hình. Chúng ta sẽ sử dụng mô hình Random Forest để dự đoán khả năng sống sót trên tàu Titanic.
        ```python
            from sklearn.ensemble import RandomForestClassifier

            # Khởi tạo mô hình
            model = RandomForestClassifier(random_state=42)

            # Huấn luyện mô hình
            model.fit(X_train, y_train)

        ```
        """)
    

    # Khởi tạo mô hình
    model = RandomForestClassifier(random_state=42)

    # Huấn luyện mô hình
    model.fit(X_train, y_train)

    st.write("🎯 Đánh giá mô hình bằng Cross-Validation")
    st.write("""
         Cross-Validation là một kỹ thuật đánh giá mô hình bằng cách chia dữ liệu thành nhiều phần, huấn luyện trên một phần và đánh giá trên phần còn lại.
             
        Để đánh giá mô hình, chúng ta sẽ sử dụng kỹ thuật Cross-Validation với 5 fold (cv=5).
        ```python
            from sklearn.model_selection import cross_val_score

            # Đánh giá mô hình bằng cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            cv_scores
        ```
        
        """)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    valid_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    st.write(f"✅ Validation Accuracy: {valid_acc:.4f}")
    st.write(f"✅ Test Accuracy: {test_acc:.4f}")

    # Hiển thị báo cáo phân loại
    st.write("📊 Classification Report (Validation):")
    st.text(classification_report(y_val, y_val_pred))
    return model, valid_acc, test_acc

def classification_report():
    
        
    X_train, X_val, X_test, y_train, y_val, y_test = phan_gioi_thieu()
    phan_train(X_train, y_train, X_val, y_val, X_test, y_test)

if __name__ == "__main__":
    classification_report()
