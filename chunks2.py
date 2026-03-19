import os
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# -------------------------------
# 1. Load API key
# -------------------------------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# -------------------------------
# 2. Embeddings (LOCAL)
# -------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# -------------------------------
# 3. Check if index exists
# -------------------------------
if os.path.exists("faiss_index"):
    print("✅ Loading existing FAISS index...")

    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

else:
    print("⚡ Creating FAISS index (first time only)...")

    # Load PDF
    loader = PyPDFLoader("ai.pdf")
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    print("Total chunks:", len(chunks))

    # Create vector store
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Save index
    vector_store.save_local("faiss_index")
    print("💾 Index saved!")

# -------------------------------
# 4. Retriever
# -------------------------------
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# -------------------------------
# 5. LLM (Gemini)
# -------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)

# -------------------------------
# 6. Chat loop
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
        You are a helpful and friendly AI assistant.

        Your job has TWO MODES:

        -----------------------------------
        MODE 1: General Conversation
        -----------------------------------
        If the user's message is any of the following:
        - Greeting (hi, hello, hey, good morning, etc.)
        - Gratitude (thanks, thank you, etc.)
        - Farewell (bye, goodbye, etc.)
        - Asking your name
        - Simple affirmation (ok, yes, sure)

        Respond naturally and politely WITHOUT using the context.

        Examples:
        - Greeting → "Hello! How can I help you today?"
        - Gratitude → "You're welcome!"
        - Farewell → "Goodbye! Have a great day!"
        - Name → "I am a chatbot."

        -----------------------------------
        MODE 2: Knowledge-based Answer (RAG)
        -----------------------------------
        If the user's question requires information:

        - Answer ONLY using the provided context.
        - Do NOT make up information.
        - If the answer is not in the context, say:
        "I couldn't find the answer in the provided information."

    {context}

    Question: {query}
    """

    # LLM response
    response = llm.invoke(final_prompt)

    print("\nBot:", response.content, "\n")