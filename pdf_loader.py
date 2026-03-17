from langchain_community.document_loaders import PyPDFLoader

# Load PDF
loader = PyPDFLoader("resume.pdf")

documents = loader.load()

# Print content
for i, doc in enumerate(documents):
    print(f"\n--- Page {i+1} ---\n")
    print(doc.page_content)