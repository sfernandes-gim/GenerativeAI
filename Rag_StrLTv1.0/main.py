import streamlit as st
import os

from chat_utility import answer_question

working_dir = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Chat with my Document",
    page_icon="📚",
    layout="centered"
)

st.title("Document Q & A - Llama 3 & Ollama")

uploaded_file =st.file_uploader(label = "Upload your file", type = ['pdf'])

user_query=st.text_input("Enter your query here")

if st.button("Get Answers"):
    bytes_data = uploaded_file.read()
    file_name=  uploaded_file.name

    #Save the files
    file_path = os.path.join(working_dir, file_name)
    with open(file_path, 'wb') as f:
        f.write(bytes_data)
    st.write("File Saved")

    answer = answer_question(file_name, user_query)

    st.success(answer)
