import os
import streamlit as st #Import Streamlit for the UI.
from dotenv import load_dotenv
from agents import consultant
from tools.search_web import search_web

load_dotenv()

# # Step 7: Build the UI
# # Page title
st.set_page_config(page_title='🦜🔗 AI research assistant')
st.title('🦜🔗 AI research assistant')
#
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input('Enter your question:'):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"Echo: {prompt}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        with st.spinner('Calculating...'):
            response = consultant.consultant_agent(prompt)
            answer = response['output']
            st.markdown(answer)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})






