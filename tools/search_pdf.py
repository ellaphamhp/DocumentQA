from langchain_text_splitters import RecursiveCharacterTextSplitter #Tool to split text into chunks.
from langchain_community.vectorstores import Chroma #Tool to create a vector store.
from langchain_openai import OpenAIEmbeddings #Tool to create embeddings from OpenAI.
from langchain.tools import tool
from tools import docs_loader
import os
from dotenv import load_dotenv
load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@tool
def search_pdf(query_text):
    """
    This function search pdf document for the most similar text to the query text related to data engineering.
    :param query_text:
    :return: List of 3 chunks of text that are most similar to the query text.
    """
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    # Step 1: Load pdf documents
    pages = docs_loader.pdf("example_docs/fundamentals-of-data-engineering.pdf")

    # Step 2: Split the documents into chunks of 1000 characters with 200 characters overlap.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(pages)

    # Step 3: Create a vector store from the document chunks.That way you can store and search for similar documents (i.e. vectors) later.
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings(openai_api_key=openai_api_key))


    # Step 4: Create a retriever that takes a query and returns the most similar document chunks.
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    docs = format_docs(retriever.get_relevant_documents(query_text))

    return docs.join("\n\n")