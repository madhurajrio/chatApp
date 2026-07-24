import os
from dotenv import load_dotenv

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

st.set_page_config(page_title="Groq ChatBot", page_icon="🤖")

st.title("🤖 LangChain + Groq Chat")

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)

    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# User input
user_input = st.chat_input("Type your message...")

if user_input:

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    human_message = HumanMessage(content=user_input)
    st.session_state.messages.append(human_message)

    # Call LLM
    response = llm.invoke(st.session_state.messages)

    # Display AI response
    with st.chat_message("assistant"):
        st.markdown(response.content)

    st.session_state.messages.append(
        AIMessage(content=response.content)
    )
