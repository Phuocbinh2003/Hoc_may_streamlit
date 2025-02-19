import streamlit as st

from buoi2.tien_su_ly_du_lieu import tien_xu_ly_du_lieu 
from buoi2.Data_Processing import report as CR


option = st.sidebar.selectbox(
    'Chọn ứng dụng:',
    ('Tiền xử lý dữ liệu','Data Processing') 
)

if option == 'Tiền xử lý dữ liệu':
    tien_xu_ly_du_lieu () 
if option == 'Data Processing':
    CR() 
else:
    st.write("Vui lòng chọn một ứng dụng từ thanh điều hướng.")
