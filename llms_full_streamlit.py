
import streamlit as st
import html
import re

with open('/home/genai/narasim_ai/practice/langchain_1/llms-full.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# Unescape HTML entities
unescaped_content = html.unescape(content)

cleaned = re.sub(r'<[^>]+>', '', unescaped_content)

st.markdown(cleaned)
