# pip install streamlit langchain langchain-openai beautifulsoup4 python-dotenv chromadb elevenlabs pybase64

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
  api_key="sk_6a8c38eb42dd1cf5a8ef0fe7c5881279e2d27474a1c64667"
)

load_dotenv()

def get_vectorstore_from_url(url):
    # Text in Dokument-Form erfassen
    loader = TextLoader('Syllabi.txt')
    document = loader.load()
    
    # Dokument in einzelne Chunks unterteilen
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    document_chunks = text_splitter.split_documents(document)
    
    # Aus den ganzen Chunks einen Vectorstore generieren
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())

    return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI(model="gpt-4o")
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Du bist ein KI-Studiengangsberater, welcher möglichen Studenten bei Fragen zum Studiengang Master Engineering and Management beantworten soll. Solltest du keine Antwort parat haben, verweise auf die Ansprechpersonen Lisa Kaiser und Ansgar Kühn. Sieze stets den User.")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain
    
def get_conversational_rag_chain(retriever_chain): 
    
    llm = ChatOpenAI(model="gpt-4o")
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Beantworte die Fragen freundlich und zuvorkommend und verwende den bereitgestellten Kontext, hier Syllabi.txt:\n\n{context}"),
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

# Funktion um jede generierte Audio-Datei instant abspielen zu lassen
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

# Funktion den Chatverlauf zurückzusetzen
def reset_chat():
    st.session_state.chat_history = [AIMessage(content="Hallo, ich bin der MEM-Bot🤖. Wie kann ich Ihnen weiterhelfen?")]
    st.session_state.response = "Hallo, ich bin der MEM-Bot. Wie kann ich Ihnen weiterhelfen?"

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
    TTS = st.checkbox("Sprachausgabe aktivieren", key="TTS")
    if TTS:
      st.info("Sprachausgabe aktiviert", icon="ℹ️")
    st.write("")
    st.write("")
    st.selectbox("Wähle eine Stimme:", ("Professor", "Student", "Darth Vader"), key = "voice")
    st.write("")
    st.write("")
    reset = st.button("Reset")
    if reset:
        reset_chat()
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

# Chat-Historie als Liste erstellen und initiativ befüllen
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hallo, ich bin der MEM-Bot🤖. Wie kann ich Ihnen weiterhelfen?"),
    ]
    st.session_state.response = "Hallo, ich bin der MEM-Bot. Wie kann ich Ihnen weiterhelfen?"

# Buttons, welche häufig gestellte Fragen beantworten
col1, col2, col3, col4 = st.columns(4)
if col1.button("Kontakt"):
    st.session_state.chat_history.append(AIMessage(content="Bei weiteren Fragen, sende gerne eine E-Mail an: mem@hs-pforzheim.de"))
    st.session_state.response = "Bei weiteren Fragen, sende gerne eine E-Mail an: mem@hs-pforzheim.de"
if col2.button("Voraussetzungen"):
    st.session_state.chat_history.append(AIMessage(content="Um zum Bewerbungsverfahren des Studiengangs Master Engineering and Management zugelassen zu werden müssen Sie einen wirtschaftsingenieurwissenschaftlichen Bachelorabschluss mit einer Mindestnote von 2,5 (gut) vorweisen können."))
    st.session_state.response = "Um zum Bewerbungsverfahren des Studiengangs Master Engineering and Management zugelassen zu werden müssen Sie einen wirtschaftsingenieurwissenschaftlichen Bachelorabschluss mit einer Mindestnote von 2,5 (gut) vorweisen können."
if col3.button("Über mich"):
    st.session_state.chat_history.append(AIMessage(content="Ich bin der MEM-Bot, Ihr persönlicher Studiengangsberater. Fragen Sie mich gerne alles was Sie wissen wollen."))
    st.session_state.response = "Ich bin der MEM-Bot, Ihr persönlicher Studiengangsberater. Fragen Sie mich gerne alles was Sie wissen wollen."


if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vectorstore_from_url("Syllabi.txt")    

# Input des Users
user_query = st.chat_input("Stelle deine Fragen hier‍ 🎓")
if user_query is not None and user_query != "":
    response = get_response(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))
    st.session_state.response = response

# conversation Text 2 Text
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
      with st.chat_message("AI"):
        st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
          st.write(message.content)

# Text 2 Speech basierend auf der Auswahl der SelectBox
if st.session_state.TTS == True:
    voice_response = client.generate(
        text = st.session_state.response,
        voice = st.session_state.voice,
        model = "eleven_multilingual_v2",
        output_format= "mp3_22050_32"
    )
    save(voice_response, "response.mp3") 
    autoplay_audio("response.mp3")
