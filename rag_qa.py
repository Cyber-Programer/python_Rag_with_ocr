import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# --- Configuration ---
PERSIST_DIRECTORY = 'chroma_db_cv'
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
load_dotenv()

# Verify API key presence
api_key = os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
if not api_key:
    raise ValueError("Error: The GOOGLE_GENERATIVE_AI_API_KEY environment variable is not set. Please set it in your .env file.")


def format_docs(docs):
    if isinstance(docs, list):
        return "\n\n".join([doc.page_content for doc in docs if hasattr(doc, "page_content")])
    return ""

def setup_rag_qa():
    # 1. Load the vector store
    print("1. Loading saved Vector Store...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # 2. Vector Store Retrieval Setup
    vector_store = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )

    # 3. LLM Setup
    print("2. Initializing Gemini LLM...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, api_key=api_key)

    # 4. Retriever Setup (Using MMR search with more results)
    # MMR = Maximum Marginal Relevance - balances relevance with diversity
    retriever = vector_store.as_retriever(
        search_type="mmr",  # Use MMR for better diversity
        search_kwargs={"k": 5, "fetch_k": 10}  # Retrieve more chunks for better context
    )

    # 5. Prompt Template
    prompt = ChatPromptTemplate.from_template("""
    You are an expert CV analysis assistant. Your task is to extract only specific pieces of information requested by the user, strictly from the provided Context.

    **CRITICAL INSTRUCTIONS:**
    1. **If the exact contact information (e.g., specific email address, full phone number, full street address) is present in the Context,** provide ONLY that information as the answer.
    2. **If the Context only contains placeholder text** like 'Contact available upon request' or 'Details are confidential', do not quote the placeholder. Instead, state clearly and concisely in a single sentence: "The specific contact details are not explicitly provided in the CV."

    Context:
    {context}

    Question: {question}""")

    # 6. RAG QA Chain Setup
    rag_chain = (
        # 1. Retrieve documents (Context)
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        # 2. Pass context and question to the prompt
        | prompt
        # 3. Pass the full prompt to the LLM
        | llm
        # 4. Parse the output into a string
        | StrOutputParser()
    )
    
    return rag_chain, retriever 


def run_rag_qa(rag_chain, retriever):
    print("\n--- RAG Q&A System Ready ---")
    print("Ask questions about the CV/Resume.")
    print("Type 'exit' or 'quit' to end the session.")
    print("-" * 30)
    
    while True:
        try:
            query = input("\nEnter your question: ")
            if query.lower() in ['exit', 'quit']:
                print("Exiting the RAG Q&A session. Goodbye!")
                break
            
            print("thinking...")
            
            # The rag_chain is designed to handle the query directly.
            # It will perform retrieval, formatting, and prompting internally.
            answer = rag_chain.invoke(query)
            
            print("\n🤖 AI Answer:")
            print(answer)
        except Exception as e:
            print(f"An error occurred: {e}")
            break

if __name__ == "__main__":
    try:
        rag_chain, retriever = setup_rag_qa()
        run_rag_qa(rag_chain, retriever)
    except Exception as e:
        print(f"A critical error occurred during setup: {e}")