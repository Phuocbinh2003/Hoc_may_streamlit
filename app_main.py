import streamlit as st

from buoi2.tien_su_ly_du_lieu import tien_xu_ly_du_lieu 
from buoi2.Data_Processing import report as CR
from buoi3.Linear_Regression import bt_buoi3

option = st.sidebar.selectbox(
    'Chọn ứng dụng:',
    ('Tiền xử lý dữ liệu','Data Processing', 'Linear Regression') 
)

if option == 'Tiền xử lý dữ liệu':
    tien_xu_ly_du_lieu () 
if option == 'Data Processing':
    CR() 
if option == 'Linear Regression':
    bt_buoi3()
else:
    st.write("Vui lòng chọn một ứng dụng từ thanh điều hướng.")
