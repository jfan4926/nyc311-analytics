from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")
response = llm.invoke("Reply in one sentence: what is SQL?")
print(response.content)
