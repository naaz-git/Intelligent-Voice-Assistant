from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Document Loaders
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import ArxivLoader
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import WebBaseLoader

# Text Splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter

# Chroma DB
from langchain_chroma import Chroma

# bot
import streamlit as st
from streamlit_mic_recorder import speech_to_text
import os
import tempfile
from gtts import gTTS
import os
import re
import shutil
import pickle

from dotenv import load_dotenv
import base64
# Handling Directory Operations
def make_empty_directory(path):
    try:
        os.makedirs(path, exist_ok=True)
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if item_path.endswith('chat_history.dump') == False:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
        print(f"All contents of '{path}' have been deleted.")
    except Exception as e:
        print(f"An error occurred: {e}")

def count_files(directory):
    print("Directory:", directory)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        return 0  # Return 0 if directory doesn't exist
    for f in os.listdir(directory):
        print("Files in directory:", f)
    return len([f for f in os.listdir(directory)])

# All document loaders
def getTextDocuments(filename):
    return TextLoader(filename).load()

def getPdfDocuments(filename):
    return PyPDFLoader(filename).load()

def getArxivDocuments(doc_code, max_docs):
    return ArxivLoader(query=doc_code, load_max_docs=max_docs).load()

def getWikiDocuments(query, max_docs):
    return WikipediaLoader(query=query, load_max_docs=max_docs).load()

# All Text Splitters

def getTextSplit(docs, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

def getTextCharSplit(docs, chunk_size, chunk_overlap):
    splitter=CharacterTextSplitter(separator="\n\n",chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

# Prepare and Get ChromaDB
def getChromsDB(documents, embeddings, path_vdb):
    if not os.path.exists(path_vdb):
        os.makedirs(path_vdb)  # Create the directory if it doesn't exist
    fileCountInDir = count_files(path_vdb)
    print(path_vdb, ":", fileCountInDir)
    if fileCountInDir <= 1:
        vectordb = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=path_vdb)
    else:
        print('Using already stored vectorDB')
        vectordb = Chroma(persist_directory=path_vdb, embedding_function=embeddings)
    return vectordb


class gblVariablesClass:
    def __init__(self):
        self.user_id = None
        self.conversation_id = None
        self.conversation_url = None
        self.conversation_title = None
        self.embeddings = None
        self.llm = None
        self.retriever = None
        self.conversation_store = None

# Function: Text-to-Speech (TTS) with Autoplay
def text_to_speech(text):
    try:
        tts = gTTS(text)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            temp_file = fp.name
        
        with open(temp_file, "rb") as audio_file:
            audio_bytes = audio_file.read()
            encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")

        autoplay_audio_html = f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{encoded_audio}" type="audio/mp3">
            Your browser does not support the audio element.
        </audio>
        """
        st.markdown(autoplay_audio_html, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred in TTS: {e}")

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

# Function: Initialize gblVariables
def initialize_variables(user_id, conversation_id, conversation_url):
    clean_user_id = re.sub(r"[^a-zA-Z0-9]", "_", user_id).lower()
    clean_conversation_id = re.sub(r"[^a-zA-Z0-9]", "_", conversation_id).lower()
    load_dotenv()
    gorq_api_key = os.getenv("GORQ_API_KEY")

    gblVariables = {
        "user_id": clean_user_id,
        "conversation_id": clean_conversation_id,
        "conversation_url": conversation_url,
        "embeddings": HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        "llm": ChatGroq(groq_api_key=gorq_api_key, model_name="Gemma2-9b-It"),
        "conversation_store": ChatMessageHistory(),
    }
    return gblVariables

# Function: Handle Voice Conversation
def handle_voice_conversation(gblVariables, prompt_flag, query):
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just reformulate it."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        gblVariables["llm"], gblVariables["retriever"], contextualize_q_prompt
    )

    # Create RAG Chain
    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        create_stuff_documents_chain(
            gblVariables["llm"],
            ChatPromptTemplate.from_messages(
                [
                    ("system", "{context}"),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            ),
        ),
    )
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda _: gblVariables["conversation_store"],
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # Execute RAG with Query
    response = conversational_rag_chain.invoke({"input": query})
    return response["answer"]

# Streamlit UI
st.title("Conversational RAG Voice Assistant")
st.markdown("Speak to the assistant and interact with the RAG system.")

if "gblVariables" not in st.session_state:
    st.session_state.gblVariables = {"conversation_store": ChatMessageHistory()}

if "conversation_store" not in st.session_state.gblVariables or st.session_state.gblVariables["conversation_store"] is None:
    st.session_state.gblVariables["conversation_store"] = ChatMessageHistory()

# Start Conversation Button
if st.button("Start Voice Conversation"):
    if st.session_state.gblVariables is None:
        st.session_state.gblVariables = initialize_variables(
            user_id="kmalhan@clarku.edu",
            conversation_id="202501152015",
            conversation_url="https://www.ibm.com/think/topics/machine-learning",
        )

st.write("**Voice Conversation Started! Speak your question now.**")

    # Speech-to-Text Interaction
    s2t_output = speech_to_text(
        language="en",
        start_prompt="⭕ TALK",
        stop_prompt="🟥 LISTENING...PRESS TO STOP",
        just_once=True,
        use_container_width=True,
        key=f"s2t_{len(st.session_state.gblVariables['conversation_store'].messages)}"
    )

    if s2t_output:
        st.write(f"**You said:** {s2t_output}")
        # Query the RAG System
        answer = handle_voice_conversation(st.session_state.gblVariables, "Answer", s2t_output)
        st.write(f"**Assistant says:** {answer}")
        # Play response as audio
        text_to_speech(answer)
    else:
        st.warning("No speech detected. Please try again.")
