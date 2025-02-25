import streamlit as st
from buoi2.tien_su_ly_du_lieu import main 
from buoi4.Classification import Classification
from buoi5.Clustering_Algorithms import ClusteringAlgorithm 
# from test import appptest as ts
option = st.sidebar.selectbox(
    'Chọn ứng dụng:',
    ('Linear Regression','Classification MNIST','Clustering Algorithms') 
)

if option == 'Linear Regression':
    main() 
if option == 'Classification MNIST':
    Classification() 
    
if option == 'Clustering Algorithms':
    ClusteringAlgorithm() 
# if option == 'test':
#     ts() 
else:
    st.write("Vui lòng chọn một ứng dụng từ thanh điều hướng.")
