import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain.chat_models import ChatOpenAI

st.title("QnA with OpenAI")

question = st.text_input("Ask what you want to know")

if question:
    llm = ChatOpenAI(model="gpt-4o", openai_api_key=os.getenv("OPENAI_KEY"))
    response = llm.invoke(question)
    st.write(response.content)
