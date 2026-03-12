import os
import re
from dotenv import load_dotenv
import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
import random

load_dotenv()

RERANK_MODEL_NAME = "ms-marco-MiniLM-L-12-v2"


@st.cache_resource
def setup_rag_pipeline():
    embeddings = OllamaEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
    )

    vector_store = Chroma(
        collection_name=os.getenv("COLLECTION_NAME"),
        embedding_function=embeddings,
        persist_directory=os.getenv("DATABASE_LOCATION"), 
    )

    base_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 20, "fetch_k": 50, "lambda_mult": 0.7}
    )

    compressor = FlashrankRerank(model=RERANK_MODEL_NAME)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
    )

    llm = ChatOllama(
        model=os.getenv("CHAT_MODEL"),
        temperature=0.2,
        keep_alive="5m"
    )
    
    return compression_retriever, llm

retriever, llm = setup_rag_pipeline()

PROMPT_TEMPLATE = """
You are an expert Old School RuneScape (OSRS) Wiki assistant.
Use the following pieces of retrieved context to answer the user's question.

Rules:
1. Answer strictly based on the context provided.
2. If the context does not contain the answer, say "I cannot find that information in the Wiki."
3. Mention the "Source" if available.

Context:
{context}

Chat History:
{chat_history}

User Question: {input}
Answer:
"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

def clean_response(response_text):
    cleaned = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
    cleaned = re.sub(r'<thinking>.*?</thinking>', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
    return cleaned.strip()

def format_history(messages):
    recent = messages[-4:]
    lines = []
    for m in recent:
        if isinstance(m, HumanMessage):
            lines.append(f"Human: {m.content}")
        elif isinstance(m, AIMessage):
            lines.append(f"AI: {m.content}")
    return "\n".join(lines)

def format_docs(docs):
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        content = doc.page_content.replace("\n", " ")
        formatted.append(f"Source: {source}\nContent: {content}")
    return "\n\n---\n\n".join(formatted)

st.set_page_config(
    page_title="OSRS Wiki RAG",
    page_icon="⚔️",
    layout="wide"
)
st.title("OSRS Wiki Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

LOADING_TEXTS = [
    "Consulting the Wise Old Man...",
    "Sifting through the Grand Exchange...",
    "Deciphering cryptic clues...",
    "Communing with the gods of Gielinor...",
    "Searching the Library of Lumbridge...",
    "Scanning the RuneScape Wiki (99/99)...",
    "Querying the database of infinite knowledge...",
    "Waiting for the servers to stop lagging...",
    "Getting 99 Computing... Be right back.",
    "Bypassing the server cache...",
    "Flipping through the Knowledge Base...",
    "Connection lost. Please wait - attempting to reestablish.",
    "🦀 $11 🦀",
    "1-tick flicking the database...",
    "Compiling a list of every Jad attempt you've ever failed...",
    "The Zulrah rotation is blue, green, blue... wait, no...",
    "The servers are physically located inside the Falador bank...",
    "Spamming spacebar...",
    "ironman btw...",
    "92 is half of 99...",
    "Warning: XP waste detected...",
    "Calculating the total time wasted on Agility rooftops...",
    "Did you talk to Oziach first?",
    "This loading text has the same drop rate as a KBD pet...",
    "Calculating the number of times you've forgotten your Dramen Staff.."
]


def get_random_loading_text() -> str:
    return random.choice(LOADING_TEXTS)

user_question = st.chat_input("Search the OSRS Wiki...")

if user_question:
    with st.chat_message("user"):
        st.markdown(user_question)
        st.session_state.messages.append(HumanMessage(user_question))

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner(get_random_loading_text()):
            retrieved_docs = retriever.invoke(user_question)
            
            if not retrieved_docs:
                full_response = "I couldn't find any relevant articles in the database."
                message_placeholder.markdown(full_response)
            else:
                context_text = format_docs(retrieved_docs)
                history_text = format_history(st.session_state.messages)
                chain = prompt | llm
                response_stream = chain.stream({
                    "context": context_text,
                    "chat_history": history_text,
                    "input": user_question
                })
                

                for chunk in response_stream:
                    full_response += chunk.content
                    message_placeholder.markdown(clean_response(full_response) + "▌")
                
                final_clean = clean_response(full_response)
                message_placeholder.markdown(final_clean)
                
                with st.expander("View Sources"):
                    for doc in retrieved_docs:
                        st.write(f"**{doc.metadata.get('source')}** (Relevance: {doc.metadata.get('relevance_score', 0):.2f})")
                        st.text(doc.page_content[:200] + "...")

    st.session_state.messages.append(AIMessage(full_response))