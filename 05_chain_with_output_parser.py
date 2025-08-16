import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
chain1 = prompt | llm
chain2 = prompt | llm | StrOutputParser()
response1 = chain1.invoke({"location": "patna",
                          "month":"april",
                          "budget":"10000"})

response2 = chain2.invoke({"location": "patna",
                          "month":"april",
                          "budget":"10000"})
print(response1)
print("Chain invocation without parser gives result of type AIMessage. With parser gives nice string in printable format")
print(response2)
