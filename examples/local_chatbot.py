import streamlit as st 
from langchain_community.llms import Ollama

with st.sidebar:

    # Select LLM
    st.subheader('Models')
    selected_model = st.sidebar.selectbox('Choose an open-source model', ['Llama2-7B', 'Mistral-7B'], key='selected_model')
    if selected_model == 'Llama2-7B':
        llm = Ollama(model="llama2") 
    elif selected_model == 'Mistral-7B':
        llm = Ollama(model="mistral") 
    
    # Clear history button
    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


# Initial message from AI
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Extend chat message
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat 
if prompt := st.chat_input():

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = llm.invoke(input=st.session_state.messages)
    msg = response
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)

