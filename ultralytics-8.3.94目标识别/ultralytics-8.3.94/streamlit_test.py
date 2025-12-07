import streamlit as st

# 显示标题
st.title("Hello, Streamlit!")

# 显示文本
st.write("这是一个简单的 Streamlit 应用。")

# 创建一个滑块
number = st.slider("选择一个数字", 0, 100, 50)

# 显示选择的数字
st.write(f"你选择的数字是: {number}")