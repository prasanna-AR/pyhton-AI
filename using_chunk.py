import os
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Use LOCAL embeddings (no API limit 🔥)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# -------------------------------
# 1. Load API key
# -------------------------------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# -------------------------------
# 2. Load PDF
# -------------------------------
loader = PyPDFLoader("ai.pdf")
documents = loader.load()

# -------------------------------
# 3. Chunk (split text)
# -------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print("Total chunks:", len(chunks))

# -------------------------------
# 4. Embeddings (LOCAL - best)
# -------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# -------------------------------
# 5. Vector Store (FAISS)
# -------------------------------
vector_store = FAISS.from_documents(chunks, embeddings)

retriever = vector_store.as_retriever()

# -------------------------------
# 6. LLM (Gemini)
# -------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)

# -------------------------------
# 7. Chat loop (User input → Answer)
# -------------------------------
print("\n📄 Chatbot Ready! Type 'exit' to quit\n")

while True:
    query = input("You: ")

    if query.lower() == "exit":
        print("Exiting chatbot...")
        break

    # Retrieve relevant chunks
    docs = retriever.invoke(query)

    context = "\n".join([doc.page_content for doc in docs])

    # Prompt
    final_prompt = f"""
    Answer the question based only on the context below:
    1. If user says "he" → reply "hello"
    2. If user says "good morning" → reply "good morning, have a great day"

    {context}

    Question: {query}
    """

    # LLM response
    response = llm.invoke(final_prompt)

    print("\nBot:", response.content, "\n")