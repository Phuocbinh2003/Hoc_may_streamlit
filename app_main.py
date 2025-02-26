import streamlit as st
from buoi2.tien_su_ly_du_lieu import main 
from buoi4.Classification import Classification
from buoi5.Clustering_Algorithms import ClusteringAlgorithms
# from test import appptest as ts
option = st.sidebar.selectbox(
    'Chọn ứng dụng:',
    ('Linear Regression','Classification MNIST','Clustering Algorithms') 
)

if option == 'Linear Regression':
    st.session_state.clear()
    
    main() 
if option == 'Classification MNIST':
    st.session_state.clear()
    
    Classification() 
    
if option == 'Clustering Algorithms':
    st.session_state.clear()
    # st.rerun()
    ClusteringAlgorithms() 
# if option == 'test':

else:
    st.write("Vui lòng chọn một ứng dụng từ thanh điều hướng.")
