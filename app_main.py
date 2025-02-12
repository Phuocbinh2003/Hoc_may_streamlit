import streamlit as st
from giai_thua import main 
option = st.sidebar.selectbox(
    'Chọn ứng dụng:',
    ('Giai thừa') 
)
if option == 'Giai thừa':
    main()  
else:
    st.write("Vui lòng chọn một ứng dụng từ thanh điều hướng.")
