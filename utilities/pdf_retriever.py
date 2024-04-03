import os

from langchain_text_splitters import RecursiveCharacterTextSplitter #Tool to split text into chunks.
from langchain_community.vectorstores import Chroma #Tool to create a vector store.
from langchain_openai import OpenAIEmbeddings #Tool to create embeddings from OpenAI.
from langchain_openai import ChatOpenAI #Tool to chat with OpenAI.
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from utilities import docs_loader

import dotenv
dotenv.load_dotenv()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def generate_response(pages, openai_api_key, query_text):
    # Step 1: Load pdf documents
    # docs = format_docs(pages)

    # Step 2: Split the documents into chunks of 1000 characters with 200 characters overlap.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(pages)

    # Step 3: Create a vector store from the document chunks.That way you can store and search for similar documents (i.e. vectors) later.
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings(openai_api_key=openai_api_key))


    # Step 4: Create a retriever that takes a query and returns the most similar document chunks.
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})


    # # Step 5: Chat with OpenAI using the RAG model.
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0,
                     openai_api_key=openai_api_key)  # Use the normal GPT-3.5 model for now

    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible. When you can, use bullet points to list the main points.
    Sign off your answer by 'WebQA', in a separate paragraph.

    {context}

    Question: {question}

    Helpful Answer:"""
    prompt = PromptTemplate.from_template(template)

    # Step 6: Create a chain that pipe together components and functions: with Context is the doc and the questions, prompt GenAI for the answer, run llm model, and Parsed output.

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain.invoke(query_text)

if __name__ == "__main__":
    pages = docs_loader.pdf("example_docs/fundamentals-of-data-engineering.pdf")
    query_text = "What are the key principles of writing on social media?"
    response = generate_response(pages=pages, openai_api_key=os.environ.get("OPENAI_API_KEY"), query_text=query_text)
    print(response)