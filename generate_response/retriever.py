from openai import OpenAI
import bs4 #Import BeautifulSoup from bs4, python's HTML parser.
from langchain_community.document_loaders import WebBaseLoader #Tool to load documents from the web.
from langchain_text_splitters import RecursiveCharacterTextSplitter #Tool to split text into chunks.
from langchain_community.vectorstores import Chroma #Tool to create a vector store.
from langchain_openai import OpenAIEmbeddings #Tool to create embeddings from OpenAI.
from langchain_openai import ChatOpenAI #Tool to chat with OpenAI.
# from langchain import hub #Tool to pull prompt template from the hub.
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def generate_response(link, openai_api_key, query_text):
    # Step 1: Load document: only keep post title, headers, and content from the full HTML.
    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    loader = WebBaseLoader(
        web_paths=(link,),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()

    # Step 2: Split the documents into chunks of 1000 characters with 200 characters overlap.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    # len(all_splits)

    # Step 3: Create a vector store from the document chunks.That way you can store and search for similar documents (i.e. vectors) later.
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings(openai_api_key=openai_api_key))

    # Test using similarity search:
    # query = "What is the difference between FAISS and ScaNN"
    # answer = vectorstore.similarity_search(query)
    # print(answer[0].page_content)

    # Step 4: Create a retriever that takes a query and returns the most similar document chunks.
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    # # query = "What is the difference between FAISS and ScaNN?"
    # # answer = retriever.invoke(query)
    # # print(answer[0].page_content)
    #
    #
    # # Step 5: Chat with OpenAI using the RAG model.
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0,
                     openai_api_key=openai_api_key)  # Use the normal GPT-3.5 model for now

    # prompt = hub.pull("rlm/rag-prompt") #This prompt template is pretty basic, but it's for RAG.
    # Or you can create your own prompt template:
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking, Ella!" at the end of the answer. However, say it in a separate paragraph at the end. Sign off your answer by 'Razi Oppa'.

    {context}

    Question: {question}

    Helpful Answer:"""
    prompt = PromptTemplate.from_template(template)
    # example_messages = prompt.invoke(
    #     {"context": "filler context", "question": "filler question"}
    # ).to_messages()
    # # print(example_messages[0].content)
    #
    # Step 6: Create a chain that pipe together components and functions: with Context is the doc and the questions, prompt GenAI for the answer, run llm model, and Parsed output.

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain.invoke(query_text)

if __name__ == "__main__":
    generate_response(link=link, openai_api_key=openai_api_key, query_text=query_text)