import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Create model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)

# Chat loop
while True:
    prompt = input("You: ")

    if prompt.lower() == "exit":
        print("Exiting chatbot...")
        break

    response = llm.invoke(prompt)

    print("\nBot:", response.content, "\n")