import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- Configuration (output from extract_pdf_data.py) ---
TEXT_FILE_PATH = 'cv_extracted_text.txt' 
PERSIST_DIRECTORY = 'chroma_db_cv' 
# English/Multilingual embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def create_vector_store():
    # 1. Load the extracted text data 
    print("1. Loading text file....")
    loader = TextLoader(TEXT_FILE_PATH, encoding='utf8')
    documents = loader.load()

    # 2. Text Splitting (chunking)
    print("2. Splitting text into chunks....")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=0,
        separators=["\n\n", "\n", " ", ""]
    )
    texts = text_splitter.split_documents(documents)
    print(f"   Number of chunks created: {len(texts)}")

    # 3. Create Embeddings
    print("3. Creating embeddings using HuggingFaceEmbeddings and vector store....")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # 4. Create and persist the vector store using Chroma
    vector_store = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    print(f"   Vector store persisted at: {PERSIST_DIRECTORY}")
    return vector_store

if __name__ == "__main__":
    if os.path.exists(TEXT_FILE_PATH):
        create_vector_store()
    else:
        print(f"Error: The file {TEXT_FILE_PATH} does not exist. Please run extract_pdf_data.py first to generate the text file.")