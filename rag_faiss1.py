import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# Load API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Step 1: Load PDF
loader = PyPDFLoader("resume.pdf")
documents = loader.load()

# Step 2: Extract text from PDF
texts = [doc.page_content for doc in documents]

# Step 3: Create embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)

# Step 4: Store in FAISS
db = FAISS.from_texts(texts, embeddings)

# Step 5: Ask question
query = "What is this PDF about?"
results = db.similarity_search(query)

# Step 6: Print answer
print("\nAnswer from PDF:\n")
print(results[0].page_content)