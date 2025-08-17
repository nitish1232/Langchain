from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

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

while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    response = runnable_chain.invoke(
        input={"input": user_input},
        config={"configurable": {"session_id":"chat123"}}
    )
    print(f"Bot: {response}")
