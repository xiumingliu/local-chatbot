from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

import streamlit as st
import utils

st.set_page_config(page_title="Search agent", page_icon="🌐")
st.header('What do your want to search for?')

class chatbot_with_search:

    def __init__(self):
        self.llm = utils.select_llm()

        with st.sidebar:
            st.sidebar.button('Clear Chat History', on_click=utils.clear_chat_history())

 
    def main(self):
        msgs = StreamlitChatMessageHistory()
        memory = ConversationBufferMemory(
            chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
        )
        if len(msgs.messages) == 0:
            msgs.clear()
            msgs.add_ai_message("How may I assist you today?")
            st.session_state.steps = {}

        avatars = {"human": "user", "ai": "assistant"}
        for idx, msg in enumerate(msgs.messages):
            with st.chat_message(avatars[msg.type]):
                # Render intermediate steps if any were saved
                for step in st.session_state.steps.get(str(idx), []):
                    if step[0].tool == "_Exception":
                        continue
                    with st.status(f"**{step[0].tool}**: {step[0].tool_input}", state="complete"):
                        st.write(step[0].log)
                        st.write(step[1])
                st.write(msg.content)

        if prompt := st.chat_input(placeholder="What is today's weather in Stockholm?"):
            st.chat_message("user").write(prompt)
            # tools = [DuckDuckGoSearchRun(name="Search")]
            tools = [TavilySearchResults()]
            chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=self.llm, tools=tools)
            executor = AgentExecutor.from_agent_and_tools(
                agent=chat_agent,
                tools=tools,
                memory=memory,
                return_intermediate_steps=True,
                handle_parsing_errors=True,
            )
            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                cfg = RunnableConfig()
                cfg["callbacks"] = [st_cb]
                response = executor.invoke(prompt, cfg)
                st.write(response["output"])
                st.session_state.steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]

if __name__ == "__main__":
    obj = chatbot_with_search()
    obj.main()