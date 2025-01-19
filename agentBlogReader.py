import streamlit as st
from streamlit_float import *

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
#from langchain_chroma import Chroma
#from langchain_chroma.vectorstores import Chroma
from langchain.vectorstores import FAISS

import os
import re
import shutil
import pickle



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
        # print(f"All contents of '{path}' have been deleted.")
    except Exception as e:
        print(f"An error occurred: {e}")

def count_files(directory):
    # print("Directory:", directory)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        return 0  # Return 0 if directory doesn't exist
    #for f in os.listdir(directory):
    #    print("Files in directory:", f)
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
    # print(path_vdb, ":", fileCountInDir)
    if fileCountInDir <= 1:
        vectordb = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=path_vdb)
    else:
        #print('Using already stored vectorDB')
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

# Main Function
def RAG(gblVariables, user_id, conversation_id, conversation_url, conversation_title, 
         prompt_flag, time_available_mins, query):

    path_folder_vdb                         = './vectorDBStore'
    path_folder_users                       = './users'

    class Config:
        arbitrary_types_allowed             = True

    # print("user_id and conversation_id", user_id, conversation_id)

    # Only LLM response
    if gblVariables == None and conversation_url == None:
        load_dotenv()
        gorq_api_key= os.getenv("GORQ_API_KEY")

        # Access the secrets

        LANGCHAIN_TRACING_V2= st.secrets[LANGCHAIN_TRACING_V2]
        LANGCHAIN_ENDPOINT=   st.secrets[LANGCHAIN_ENDPOINT]
        LANGCHAIN_API_KEY=    st.secrets["LANGCHAIN_API_KEY"]
        LANGCHAIN_PROJECT=    st.secrets[LANGCHAIN_PROJECT]

        GORQ_API_KEY=   st.secrets["GORQ_API_KEY"]
        HF_TOKEN =      st.secrets["HF_TOKEN"]
        HUGGINGFACEHUB_API_TOKEN=     st.secrets[HUGGINGFACEHUB_API_TOKEN]
        SERP_API_KEY=                 st.secrets[SERP_API_KEY]
             
        llm         = ChatGroq(groq_api_key=gorq_api_key, model_name="Gemma2-9b-It")
        response    = llm.invoke("Respond only in English language. " + query)
        return response
    if gblVariables == None:
        # Try to fetch from local store
        clean_user_id                       = re.sub(r'[^a-zA-Z0-9]', '_', user_id).lower()
        clean_conversation_id               = re.sub(r'[^a-zA-Z0-9]', '_', conversation_id).lower()
        folder_name                         = path_folder_users + clean_user_id
        count_files(folder_name)
        gblVariablesStore                   = folder_name + '/' + clean_conversation_id + '.pkl'
        try:
            if os.path.exists(gblVariablesStore):
                with open(gblVariablesStore, "rb") as file:
                    gblVariables            = pickle.load(file)
                    #gblVariables            = dill.load(file)
                    gblVariables.llm        = ChatGroq(groq_api_key=gorq_api_key, model_name="Gemma2-9b-It")
            else:
                raise FileNotFoundError(f"The file '{gblVariablesStore}' does not exist.")
        except:
            # Nothhing works
            # To be initialized only Once
            gblVariables                    = gblVariablesClass()
            gblVariables.user_id            = user_id
            gblVariables.conversation_id    = conversation_id
            gblVariables.conversation_url   = conversation_url
            gblVariables.conversation_title = conversation_title

            load_dotenv()
            gorq_api_key                    = os.getenv("GORQ_API_KEY")
            os.environ['HF_TOKEN']          = os.getenv("HF_TOKEN")
            os.environ["USER_AGENT"]        = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

            gblVariables.embeddings         = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            gblVariables.llm                = ChatGroq(groq_api_key=gorq_api_key, model_name="Gemma2-9b-It")

        # print("::gblVariables.embeddings::", gblVariables.embeddings)
        # print("::gblVariables.llm::", gblVariables.llm)

        path_vdb                        = path_folder_vdb + clean_user_id + '/' + clean_conversation_id + '/'
        count_files(path_vdb)
        documents                       = []
        loader                          = WebBaseLoader(conversation_url)
        docs                            = loader.load()
        documents.extend(docs)
        chunks                          = getTextSplit(documents, 4096, 256)
        vectorstore                     = getChromsDB(chunks, gblVariables.embeddings, path_vdb)
        gblVariables.retriever          = vectorstore.as_retriever()
        gblVariables.conversation_store = ChatMessageHistory()

        ####### Dump Start
        tmp_llm                = gblVariables.llm
        tmp_retriever          = gblVariables.retriever
        gblVariables.llm       = None
        gblVariables.retriever = None
        with open(gblVariablesStore, 'wb') as file:
            pickle.dump(gblVariables, file)
            #dill.dump(gblVariables, file)
        gblVariables.llm       = tmp_llm
        gblVariables.retriever = tmp_retriever
        ####### Dump End

    # History 1. System Prompt Declaration
    contextualize_q_system_prompt   = (
        "Given a chat history and the latest user question"
        "which will have reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    # History 2. Clubbed Final Prompt Declaration
    contextualize_q_prompt          = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    # History 3. Retriever Declaration
    history_aware_retriever         = create_history_aware_retriever(gblVariables.llm, 
                                                        gblVariables.retriever,
                                                        contextualize_q_prompt)

    # Query 1. System Prompt Declaration
    # if prompt_flag == 'Answer':
    system_prompt = (
        "You are in conversation with a human, with the primary goal of answering their questions. Use the uploaded "
        "documents as the main source of information, but supplement with your base knowledge if the documents don't "
        "cover the query, explicitly stating so. Respond courteously to greetings like 'Hello' or 'Good morning.' "
        "Respond only in English language. "
        "If you don't know an answer, acknowledge it. Don't add anything before or after your response. "
        "Keeping responses in paragraphs with "
        "150-200 words, provide answer to given question: "
        "\n\n"
        "{context}"
        )
    if prompt_flag == 'Questions':
        system_prompt = (
            "Your primary task is to summarize the content provided in the documents and engage in a conversational "
            "style of multiple questions and answers. The goal is to ensure that, by the end of the conversation, "
            "the human understands the full content. Use the uploaded documents as the main source of information, "
            "supplementing with your base knowledge if the documents do not cover the query, and explicitly mention "
            "when this is the case. Respond courteously to greetings such as 'Hello' or 'Good morning.' "
            "Respond only in English language. "
            "If you are "
            "unsure of an answer, acknowledge it honestly. At this stage try to formulate questions only that if "
            "anwered will cover full understanding of content in documents. Make minimum "
            + str(time_available_mins*80//100) +
            " questions and maximum "
            + str(time_available_mins*120//100) +
            " questions, "
            "ensuring each question is clear, concise, and contributes to a comprehensive understanding of the content. "
            "Don't add anything before or after list of questions"
            "\n\n"
            "{context}"
            )
    elif prompt_flag == 'Summarize':
        system_prompt = (
            "Your main job is summarize the content added as document, with the primary goal of sharing important information from document. Use the uploaded "
            "documents as the main source of information, but supplement with your base knowledge. "
            "Respond courteously to greetings like 'Hello' or 'Good morning.' "
            "Respond only in English language. "
            "If find it difficult to summarize, share that. Make minimum "
            + str(time_available_mins*80//100) +
            " paragraphs and maximum "
            + str(time_available_mins*120//100) +
            " paragraphs, Keeping 300-400 words in each paragraph."
            "\n\n"
            "{context}"
            )
    # Query 2. Clubbed Final Prompt Declaration
    qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
    
    # Query 3. Retriever Chain Declaration
    question_answer_chain           = create_stuff_documents_chain(gblVariables.llm, qa_prompt)

    # History + Query Chain Declaration
    rag_chain                       = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session_id:str)->BaseChatMessageHistory:
        if gblVariables.conversation_store is None:
            gblVariables.conversation_store  = ChatMessageHistory()
        return gblVariables.conversation_store
    
    # Message Format Definition
    conversational_rag_chain        = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key          = "input",
        history_messages_key        = "chat_history",
        output_messages_key         = "answer"
    )

    # Execution of RAG
    response                        = conversational_rag_chain.invoke(
        {"input": query},
        config={
            "configurable": {"session_id":gblVariables.user_id+gblVariables.conversation_id}
        },
    )
    return (gblVariables, response['answer'])


###### Questions
def getQuestionsFromBlog(conversation_id, conversation_url, conversation_title, time_available_mins):
    gblVariables        = None
    user_id             = 'kmalhan@clarku.edu'
    prompt_flag         = 'Questions'    # Questions/Answer/Summarize
    query               = 'Make questions that can be answered from document shared'    # Only required in case of Answer
    gblVariables, questions = RAG(gblVariables, user_id, conversation_id, conversation_url, conversation_title, 
                                prompt_flag, time_available_mins, query)
    print("OUT questions", questions)

    newoutput = RAG(None, user_id, conversation_id, None, None, 
            None, 1, 'Check below questions and convert them into an python string list as variable name python_questions:\n' + questions)
    newoutput = newoutput.content.split("\n")
    questionList = []
    for obj in newoutput:
        try:
            matches = re.findall(r'"((?:\\.|[^"\\])*)"', obj)[0]
            questionList.append(matches)
        except:
            pass
    print("questionList", questionList)
    return (gblVariables, questionList)

'''
###### General LLM
user_id             = 'kmalhan@clarku.edu'
conversation_id     = '202501152015'
response            = RAG(None, user_id, conversation_id, None, None, 
                               None, 1, 'Hi How are you?')
print("General LLM", response)


conversation_id     = '202501190315'
conversation_url    = 'https://www.ibm.com/think/topics/machine-learning'
conversation_title  = ''
time_available_mins = 10    # Only required in case of Questions and Summarize
gblVariables, questionList = getQuestionsFromBlog(conversation_id, conversation_url, conversation_title, time_available_mins)


answerList = []
for question in questionList:
    gblVariables, answer = RAG(gblVariables, None, None, None, None, 
                               'Answer', 0, question)
    answerList.append(answer)
print("answerList", answerList)
'''
'''
###### Answer
gblVariables        = None
user_id             = 'kmalhan@clarku.edu'
conversation_id     = '202501152015'
conversation_url    = 'https://www.ibm.com/think/topics/machine-learning'
conversation_title  = ''
prompt_flag         = 'Answer'    # Questions/Answer/Summarize
time_available_mins = 0    # Only required in case of Questions and Summarize
query               = 'What are the key differences between machine learning, deep learning, and neural networks?'    # Only required in case of Answer
gblVariables, answer = RAG(gblVariables, user_id, conversation_id, conversation_url, conversation_title, 
                               prompt_flag, time_available_mins, query)
print("OUT answer", answer)

###### Summarize
gblVariables        = None
user_id             = 'kmalhan@clarku.edu'
conversation_id     = '202501152015'
conversation_url    = 'https://www.ibm.com/think/topics/machine-learning'
conversation_title  = ''
prompt_flag         = 'Summarize'    # Questions/Answer/Summarize
time_available_mins = 10    # Only required in case of Questions and Summarize
query               = 'Make Summary'    # Only required in case of Answer
gblVariables, summary = RAG(gblVariables, user_id, conversation_id, conversation_url, conversation_title, 
                               prompt_flag, time_available_mins, query)
print("OUT summary", summary)
'''
