import streamlit as st
option = st.sidebar.selectbox(
    'Chọn ứng dụng:',
    ('Giai thừa') 
)
if option == 'Giai thừa':
    run_app1()  
else:
    st.write("Vui lòng chọn một ứng dụng từ thanh điều hướng.")
