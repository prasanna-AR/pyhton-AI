import os
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Use LOCAL embeddings (no API limit )

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

# 💾 Save index to disk
vector_store.save_local("faiss_index")

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

    # Better conversation handling
    greetings = ["hi", "hello", "hey"]
    thanks = ["thanks", "thank you"]
    bye = ["bye", "goodbye"]

    if any(word in query.lower() for word in greetings):
        print("\nBot: Hey there! How can I help you? 😊\n")
        continue

    elif any(word in query.lower() for word in thanks):
        print("\nBot: You're welcome! 😄\n")
        continue

    elif any(word in query.lower() for word in bye):
        print("\nBot: Bye! Have a great day 🚀\n")
        break

    # 🔍 RAG retrieval
    docs = retriever.invoke(query)

    # If no useful docs → fallback to normal AI
    if not docs:
        response = llm.invoke(query)
        print("\nBot:", response.content, "\n")
        continue

    # 🧠 Build context
    context = "\n".join([doc.page_content for doc in docs])

    # ✅ Clean prompt (no confusion)
    final_prompt = f"""
You are a helpful assistant.

Use the context below if it is useful.
If the context is not relevant, answer using your own knowledge.

Context:
{context}

Question: {query}
"""

    response = llm.invoke(final_prompt)

    print("\nBot:", response.content, "\n")