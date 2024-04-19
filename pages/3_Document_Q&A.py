import os
import tempfile

import streamlit as st
import utils

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.base import BaseCallbackHandler



st.set_page_config(page_title="Doc Q&A", page_icon="📄")
st.header('Which document would you like me to read?')
st.write('Upload a document and ask me any questions about it.')



class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)



class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")


class chatbot_with_doc:

    def __init__(self):
        self.llm = utils.select_llm()

        with st.sidebar:
            st.sidebar.button('Clear Chat History', on_click=utils.clear_chat_history())


    
    #@st.cache_resource(ttl="1h")
    @st.spinner('Analyzing documents..')
    def configure_retriever(self, uploaded_files):
        # Read documents
        docs = []
        temp_dir = tempfile.TemporaryDirectory()
        for file in uploaded_files:
            temp_filepath = os.path.join(temp_dir.name, file.name)
            with open(temp_filepath, "wb") as f:
                f.write(file.getvalue())
            loader = PyPDFLoader(temp_filepath)
            docs.extend(loader.load())

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # Create embeddings and store in vectordb
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)

        # Define retriever
        retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})

        return retriever
    
 
    def main(self):

        uploaded_files = st.sidebar.file_uploader(
            label="Upload PDF files", type=["pdf"], accept_multiple_files=True
        )
        if not uploaded_files:
            st.info("Please upload PDF documents to continue.")
            st.stop()

        retriever = self.configure_retriever(uploaded_files)

        # Setup memory for contextual conversation
        msgs = StreamlitChatMessageHistory()
        memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

        qa_chain = ConversationalRetrievalChain.from_llm(
            self.llm, retriever=retriever, memory=memory, verbose=True
        )

        if len(msgs.messages) == 0:
            msgs.clear()
            msgs.add_ai_message("How can I help you?")

        avatars = {"human": "user", "ai": "assistant"}
        for msg in msgs.messages:
            st.chat_message(avatars[msg.type]).write(msg.content)


        if user_query := st.chat_input(placeholder="Ask me anything!"):
            st.chat_message("user").write(user_query)

            with st.chat_message("assistant"):
                retrieval_handler = PrintRetrievalHandler(st.container())
                stream_handler = StreamHandler(st.empty())
                response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])

if __name__ == "__main__":
    obj = chatbot_with_doc()
    obj.main()