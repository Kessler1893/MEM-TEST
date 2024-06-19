# pip install streamlit langchain langchain-openai beautifulsoup4 python-dotenv chromadb elevenlabs

from elevenlabs.client import ElevenLabs
from elevenlabs import play, save
import pybase64
import os
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import TextLoader

client = ElevenLabs(
  api_key="sk_348505f1ab3de49c73e9d72a4c591be44abf513a54e14796"
)

load_dotenv()

def get_vectorstore_from_url(url):
    # get the text in document form
    loader = TextLoader('Syllabi.txt')
    document = loader.load()
    
    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    document_chunks = text_splitter.split_documents(document)
    
    # create a vectorstore from the chunks
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())

    return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Du bist ein KI-Studiengangsberater, welcher möglichen Studenten bei Fragen zum Studiengang Master Engineering and Management beantworten soll. Solltest du keine Antwort parat haben, verweise auf die Ansprechpersonen Lisa Kaiser und Ansgar Kühn.")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain
    
def get_conversational_rag_chain(retriever_chain): 
    
    llm = ChatOpenAI()
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Beantworte die Fragen freundlich und zuvorkommend und verwende den bereitgestellten Kontext, hier Syllabi.txt. Falls du die Frage nicht beantworten kannst verweise auf Lisa.Kaiser@hs-pforzheim.de:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']

def autoplay_audio(file_path: str):
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = pybase64.b64encode(data).decode()
            md = f"""
                <audio controls autoplay="true">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
                """
            st.markdown(
                md,
                unsafe_allow_html=True,
            )

# app config
st.set_page_config(page_title="MEM-Bot", page_icon="🤖")
st.title("MEM-Bot 📚")

# sidebar
with st.sidebar:
    #st.logo("logo.svg")
    #st.sidebar.image("logo.svg", width=100)
    st.header("Hochschule Pforzheim - Master Engineering and Management M. Sc.")
    st.write("")
    st.write("")
    TTS = st.checkbox("Sprachausgabe aktivieren")
    if TTS:
      st.info("Sprachausgabe aktiviert", icon="ℹ️")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.info("Wichtige Links", icon="ℹ️")
    st.link_button("Zur MEM Seite", "https://engineeringpf.hs-pforzheim.de/master/wirtschaftsingenieurwesen/engineering_and_management")
    st.link_button("Mail an Studiengangsassistenz", "mailto:lisa.kaiser@hs-pforzheim.de")

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hallo, ich bin der MEM-Bot🤖. Wie kann ich dir weiterhelfen?"),
    ]
  
if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vectorstore_from_url("Syllabi.txt")    

    # user input
user_query = st.chat_input("Stelle deine Fragen hier‍ 🎓")
if user_query is not None and user_query != "":
    response = get_response(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))

# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
      with st.chat_message("AI"):
        if TTS:
          #Text 2 Speech
          voice_response = client.generate(
            text = message.content,
            voice = "PeterMeter",
            model = "eleven_multilingual_v2",
            output_format= "mp3_22050_32"
          )
          save(voice_response, "response.mp3") 
          autoplay_audio("response.mp3")
         
        #Text 2 Text
        st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
          st.write(message.content)
  
  
