import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

load_dotenv()
llm = ChatOpenAI(model="gpt-4o", openai_api_key=os.getenv("OPENAI_KEY"))
prompt = ChatPromptTemplate(
    [
        ("system", "you are a helpful assistant. Answer the given question."),
        ("human", "{question}")
    ]
)
chain = prompt | llm | StrOutputParser()

st.title("ChatPromptTemplate Examples")
question = st.text_input("What do you want to know")
if question:
    st.write(
        chain.invoke({"question": question})
    )
