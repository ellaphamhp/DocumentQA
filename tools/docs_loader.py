from langchain_community.document_loaders import PyPDFLoader

def pdf(link):
    loader = PyPDFLoader(link)
    pages = loader.load_and_split()
    return pages


