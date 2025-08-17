from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
import streamlit as st
import uuid

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", openai_api_key=os.getenv("OPENAI_KEY"))

prompt = ChatPromptTemplate(
    [
        ("system", "You are a helpful assistant"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ]
)
chain = prompt | llm | StrOutputParser()

def get_session_history(session_id):
    return FileChatMessageHistory(file_path=f"{session_id}.json")


runnable_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)


st.set_page_config(page_title="LLM Chatbot", layout="centered")
st.title("Chat with LLM")
session_id = ""

if "session_id" not in st.session_state:
    session_id = uuid.uuid4()
    st.session_state.session_id = session_id

user_input = st.chat_input("Type your message here...")
session_history = get_session_history(session_id)
for msg in session_history.messages:
    role = "human" if isinstance(msg, HumanMessage) else "ai"
    with st.chat_message(role):
        st.markdown(msg.content)

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Loading..."):
        response = runnable_chain.invoke(
            input={"input": user_input},
            config={"configurable": {"session_id":session_id}}
        )

    with st.chat_message("ai"):
        st.markdown(response)
