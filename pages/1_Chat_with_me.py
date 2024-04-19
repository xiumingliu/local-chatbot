import utils
import streamlit as st

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

st.set_page_config(page_title="Chatbot", page_icon="💬")
st.header('What do you want to talk about?')
st.write('Feel free to ask me about anything.')

class chatbot:

    def __init__(self):
        self.llm = utils.select_llm()

        with st.sidebar:
            st.sidebar.button('Clear Chat History', on_click=utils.clear_chat_history())

    def main(self):
 
        msgs = StreamlitChatMessageHistory()

        if len(msgs.messages) == 0:
            msgs.clear()
            msgs.add_ai_message("How can I help you?")

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an AI chatbot having a conversation with a human."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        chain = prompt | self.llm


        chain_with_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: msgs,  # Always return the instance created earlier
            input_messages_key="question",
            history_messages_key="history",
        )

        avatars = {"human": "user", "ai": "assistant"}
        for msg in msgs.messages:
            st.chat_message(avatars[msg.type]).write(msg.content)


        if prompt := st.chat_input():
            st.chat_message("human").write(prompt)

            # As usual, new messages are added to StreamlitChatMessageHistory when the Chain is called.
            config = {"configurable": {"session_id": "any"}}
            response = chain_with_history.stream({"question": prompt}, config)
            st.chat_message("ai").write(response)

if __name__ == "__main__":
    obj = chatbot()
    obj.main()