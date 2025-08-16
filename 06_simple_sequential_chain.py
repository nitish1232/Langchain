import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o", openai_api_key=os.getenv("OPENAI_KEY"))
def get_title(title):
    st.write(title)
    return title

prompt1 = PromptTemplate(
    input_variables=["topic"],
    template="""
        You are an exper speech writer.
        Provide an impactful speech title for the topic {topic}.
        Answer most best one title.
    """
)
chain1 = prompt1 | llm |  StrOutputParser() | get_title

prompt2 = PromptTemplate(
    input_variables=["title"],
    template="""
        You are an excellent speech writer. Create a 300 word speech for the given {topic}.
    """
)
chain2 = prompt2 | llm

final_chain = chain1 | chain2 | StrOutputParser()

st.title("Speech Generator")
topic = st.text_input("Enter a topic")
if topic:
    st.write(
        final_chain.invoke({"topic": topic})
    )
