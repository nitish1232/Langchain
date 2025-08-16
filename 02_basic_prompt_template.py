import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

st.write("Speech Creator")
topic = st.text_input("Enter a topic you want to create speech for")

if topic:
    llm = ChatOpenAI(model="gpt-4o", openai_api_key=os.getenv("OPENAI_KEY"))
    prompt = PromptTemplate(
        input_variables=["topic"],
        template="You are an expert assistant. Create short speech of about 2 paragraphs on the topic: {topic}"
    )
    response = llm.invoke(prompt.format(topic=topic))
    st.write(response.content)
