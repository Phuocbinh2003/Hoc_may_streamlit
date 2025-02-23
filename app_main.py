import streamlit as st

from buoi2.tien_su_ly_du_lieu import main 
from buoi2.Data_Processing import report as CR
from buoi3.Linear_Regression import bt_buoi3
from buoi4.Classification import Classification
option = st.sidebar.selectbox(
    'Chọn ứng dụng:',
    ('Linear Regression','Classification') 
)

if option == 'Linear Regression':
    main() 
if option == 'Classification':
    Classification() 
# if option == 'Data Processing':
#     CR() 
# if option == 'Linear Regression':
#     bt_buoi3()
else:
    st.write("Vui lòng chọn một ứng dụng từ thanh điều hướng.")
