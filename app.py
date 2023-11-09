import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
from llama_index.llms import OpenAI
import openai


st.set_page_config(page_title="Chat with docs, Powered by llamaindex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = "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"

st.title("Chat with docs, Powered by llamaindex")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about the docs"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing data"):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        print(docs)
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert on the Wikipedia and your job is to answer questions. Assume that all questions are related to the Wikipedia articles. Keep your answers contextual and based on facts – do not hallucinate features."))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index
    

index = load_data()

if "chat_engine" not in st.session_state.keys():
    st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)
