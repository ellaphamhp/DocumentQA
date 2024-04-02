import bs4 #Import BeautifulSoup from bs4, python's HTML parser.
from langchain_community.document_loaders import WebBaseLoader #Tool to load documents from the web.

# def web_loader(link):
#     loader = WebBaseLoader(link)
#     docs = loader.load()
#     return(docs[0].page_content)
#
from langchain_community.document_loaders import PyPDFLoader

def pdf_loader(link):
    loader = PyPDFLoader(link)
    pages = loader.load_and_split()
    return pages


