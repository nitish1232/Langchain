from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import os
from langchain.vectorstores import FAISS

load_dotenv()


text_loader = TextLoader("12_sample_text.txt")
documents = text_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

embeddings_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_KEY"))
vectorstore = FAISS.from_documents(chunks, embeddings_model)

query = "What is Jupiter made of?"
results = vectorstore.similarity_search(query, k=3)
for doc in results:
    print(doc.page_content)


retriever = vectorstore.as_retriever()
print(retriever.get_relevant_documents("Brief me about this doc in 3 sentences"))
