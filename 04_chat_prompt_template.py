import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

st.write("Travel guide")
location = st.text_input("Enter a place to visit")
month = st.selectbox("Select month of visit", ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug",
                                               "Sept", "Oct", "Nov", "Dec"])
budget = st.slider("How much you can spend (in INR)", min_value=100000, max_value=1000000)

if location and month and budget:
    llm = ChatOpenAI(model="gpt-4o", openai_api_key=os.getenv("OPENAI_KEY"))
    human_prompt = """
        You are a travel guide. You have to suggest users with following information when they
        select a {location}.
        1. Is it best time to visit during {month} 
        2. Famous sight seeing locations.
        3. Famous local foods to try.
        4. Hotels and restaurants within budget {budget}
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="You are a helpful travel planner."),
            ("human", human_prompt)
        ]
    )
    response = llm.invoke(prompt.format(
        location=location,
        month=month,
        budget=budget
    ))
    st.write(response.content)
