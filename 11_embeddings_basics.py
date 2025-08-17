from dotenv import load_dotenv
import os
from langchain.embeddings import OpenAIEmbeddings

load_dotenv()
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_KEY"))

query = "Hello World, from embeddings"
emb = embeddings.embed_query(query)
print(emb)
print(len(emb))
