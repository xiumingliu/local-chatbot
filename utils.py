import streamlit as st
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from dotenv import load_dotenv
import os


#decorator
def enable_chat_history(func):

    # to clear chat history after swtching chatbot
    current_page = func.__qualname__
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = current_page
    if st.session_state["current_page"] != current_page:
        try:
            st.cache_resource.clear()
            del st.session_state["current_page"]
            del st.session_state["messages"]
        except:
            pass

    # to show chat history on ui
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    def execute(*args, **kwargs):
        func(*args, **kwargs)
    return execute

# Select LLM on the sidebar
def select_llm():
    with st.sidebar:

        # Select LLM
        st.subheader('Models')
        selected_model = st.sidebar.selectbox('Select a language model', ['Llama2-7B', 
                                                                              'Mistral-7B',
                                                                              'GPT-3.5 Turbo',
                                                                              'GPT-4'], key='selected_model')
        #if selected_model == 'Llama2-7B':
        #    llm = Ollama(model="llama2") 
        #elif selected_model == 'Mistral-7B':
        #    llm = Ollama(model="mistral")

        match selected_model:
            case 'Llama2-7B':
                llm = Ollama(model="llama2") 
            case 'Mistral-7B': 
                llm = Ollama(model="mistral")
            case 'GPT-3.5 Turbo': 
                load_dotenv()
                OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
                TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
                llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, streaming=True)
            case 'GPT-4': 
                load_dotenv()
                OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
                TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
                llm = ChatOpenAI(model="gpt-4", temperature=0, streaming=True)


    return llm

# Clear history button on the sidebar
def clear_chat_history():
    msgs = StreamlitChatMessageHistory()
    st.session_state.steps = {} 

    msgs.clear()
    msgs.add_ai_message("How may I assist you today?")



def display_msg(msg, author):
    """Method to display message on the UI

    Args:
        msg (str): message to display
        author (str): author of the message -user/assistant
    """
    st.session_state.messages.append({"role": author, "content": msg})
    st.chat_message(author).write(msg)