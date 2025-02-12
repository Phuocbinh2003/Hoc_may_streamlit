import streamlit as st
from giai_thua import main as GT
option = st.sidebar.selectbox(
    'Chọn ứng dụng:',
    ('Giai thừa') 
)
if option == 'Giai thừa':
    GT()  
else:
    st.write("Vui lòng chọn một ứng dụng từ thanh điều hướng.")
