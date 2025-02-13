import streamlit as st
from giai_thua import main as GT
from buoi2.tien_su_ly_du_lieu import tien_xu_ly_du_lieu as TXLDL
option = st.sidebar.selectbox(
    'Chọn ứng dụng:',
    ('Giai thừa','Tiền xử lý dữ liệu') 
)
if option == 'Giai thừa':
    GT()
if option == 'Giai thừa':
    TXLDL() 
else:
    st.write("Vui lòng chọn một ứng dụng từ thanh điều hướng.")
