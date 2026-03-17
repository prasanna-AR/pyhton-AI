from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader

# Load TXT files
text_loader = DirectoryLoader(
    "data",
    glob="*.txt",
    loader_cls=TextLoader
)

# Load PDF files
pdf_loader = DirectoryLoader(
    "data",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

# Load documents
documents = text_loader.load() + pdf_loader.load()

# Print content
for i, doc in enumerate(documents):
    print(f"\n--- Document {i+1} ---\n")
    print(doc.page_content)