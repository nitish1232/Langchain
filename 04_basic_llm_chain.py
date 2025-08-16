import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

st.write("Travel guide")
location = st.text_input("Enter a place to visit")
month = st.selectbox("Select month of visit", ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug",
                                               "Sept", "Oct", "Nov", "Dec"])
budget = st.number_input("How much you can spend (in INR)")

if location and month and budget:
    llm = ChatOpenAI(model="gpt-4o", openai_api_key=os.getenv("OPENAI_KEY"))
    prompt = PromptTemplate(
        input_variables=["location", "month", "budget"],
        template="""
            You are a travel guide. You have to suggest users with following information when they
            select a {location}.
            1. Is it best time to visit during {month} 
            2. Famous sight seeing locations.
            3. Famous local foods to try.
            4. Hotels and restaurants within budget {budget}
        """
    )

    # This is the chain creation format in langchain which is called LCEL (LangChain Expression Language)
    chain = prompt | llm
    response =  chain.invoke({"location": location,
                           "month":month,
                           "budget":budget})
    st.write(response.content)
