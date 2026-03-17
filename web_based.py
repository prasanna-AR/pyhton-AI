from langchain_community.document_loaders import WebBaseLoader
import os

os.environ["USER_AGENT"] = "my-langchain-app"

url = "https://en.wikipedia.org"

loader = WebBaseLoader(url)
documents = loader.load()

for i, doc in enumerate(documents):
    print(f"\n--- Document {i+1} ---\n")
    print(doc.page_content[:500])