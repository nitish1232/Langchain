import os
from dotenv import load_dotenv
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
llm = ChatOpenAI(model="gpt-4o", openai_api_key=os.getenv("OPENAI_KEY"))


def get_title(title):
    st.write(title)
    return title

def get_input_for_chain_2(title):
    print(f"Title:{title}, Emotion:{emotion}")
    return {"title": title, "emotion": emotion}


prompt1 = PromptTemplate(
    input_variables=["topic"],
    template="""
        You are an exper speech writer.
        Provide an impactful speech title for the topic {topic}.
        Answer most best one title.
    """
)
chain1 = prompt1 | llm | StrOutputParser() | get_title

prompt2 = PromptTemplate(
    input_variables=["title", "emotion"],
    template="""
        You are an excellent speech writer. Create a 300 word speech for the given {title}
        and you will consider the speech content based on the emotion {emotion}.
    """
)
chain2 = prompt2 | llm

st.title("Speech Generator")
topic = st.text_input("Enter a topic")
emotion = st.text_input("Enter an emotion")
if topic and emotion:
    final_chain = chain1 | get_input_for_chain_2 | chain2 | StrOutputParser()
    st.write(final_chain.invoke(
        {"topic": topic, "emotion": emotion}
    ))
