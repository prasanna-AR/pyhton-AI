import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

texts = [
    "AI is the simulation of human intelligence",
    "Machine learning is a subset of AI"
]

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)

db = FAISS.from_texts(texts, embeddings)

query = "What is AI?"
results = db.similarity_search(query)

print(results[0].page_content)