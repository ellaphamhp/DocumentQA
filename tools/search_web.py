from langchain_exa import ExaSearchRetriever, TextContentsOptions
from langchain.tools import tool
from dotenv import load_dotenv
load_dotenv()

def format_articles(articles): #Function to join all the chunks retrieved from the documents
    return "\n\n".join(article.page_content +' Source (url):'+ article.metadata["url"] for article in articles)

@tool
def search_web(query_text):
    '''
    This function search web for article most relevant to the query text.
    :return:
    '''
    # retrieve 3 documents, with content truncated at 1000 characters
    exaretriever = ExaSearchRetriever(
        k=3,
        text_contents_options=TextContentsOptions(max_length=300),
        start_published_date="2023-01-01",
    )

    articles = format_articles(exaretriever.get_relevant_documents(query_text))

    return articles.join("\n\n")

