import streamlit as st
from buoi2.tien_su_ly_du_lieu import main 
from buoi4.Classification import Classification
option = st.sidebar.selectbox(
    'Chọn ứng dụng:',
    ('Linear Regression','Classification') 
)

if option == 'Linear Regression':
    main() 
if option == 'Classification':
    Classification() 

else:
    st.write("Vui lòng chọn một ứng dụng từ thanh điều hướng.")
