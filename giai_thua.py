import streamlit as st
import math

def fact(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * fact(n - 1)

def main():
    st.title("Ứng dụng tính giai thừa")
    
    # Nhập số nguyên
    n = st.number_input("Nhập số nguyên n:", min_value=0, step=1, value=1)
    
    if st.button("Tính giai thừa"):
        result = fact(n)
        st.write(f"Giai thừa của {n} là: {result}")

if __name__ == "__main__":
    main()
