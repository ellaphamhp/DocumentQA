import os
import streamlit as st #Import Streamlit for the UI.
from dotenv import load_dotenv
from generate_response import retriever

load_dotenv()


# # Step 7: Build the UI
# # Page title
st.set_page_config(page_title='🦜🔗 Ask the Doc App')
st.title('🦜🔗 Ask the Doc App')
#
# File upload
link = "https://lilianweng.github.io/posts/2023-06-23-agent/"
# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not link)

# Form input and query
result = []


with st.form('myform', clear_on_submit=True):
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    submitted = st.form_submit_button('Submit', disabled=not(link and query_text))

    if submitted:
        with st.spinner('Calculating...'):
            response = retriever.generate_response(link, openai_api_key, query_text)
            result.append(response)

if len(result):
    st.info(response)
