import os
from dotenv import load_dotenv
import streamlit as st
from langchain.agents import AgentExecutor
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
import re

load_dotenv()

print(f"Current working directory: {os.getcwd()}")
print(f"Python script location: {__file__}")
env_files = ['.env', '../.env', '../../.env']
for env_file in env_files:
    if os.path.exists(env_file):
        print(f"Found .env file at: {os.path.abspath(env_file)}")
        with open(env_file, 'r') as f:
            lines = f.readlines()[:10]
            print(f"Content preview:")
            for line in lines:
                if 'CHAT_MODEL' in line:
                    print(f"  {line.strip()}")

print(f"CHAT_MODEL: {os.getenv('CHAT_MODEL')}")
print(f"EMBEDDING_MODEL: {os.getenv('EMBEDDING_MODEL')}")

embeddings = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
)

vector_store = Chroma(
    collection_name=os.getenv("COLLECTION_NAME"),
    embedding_function=embeddings,
    persist_directory=os.getenv("DATABASE_LOCATION"), 
)

llm = ChatOllama(
    model=os.getenv("CHAT_MODEL"),
    temperature=0
)


prompt = PromptTemplate.from_template("""                                
You are a helpful Old school runescape assistant. You will be provided with a query and a chat history.
Your task is to retrieve relevant information from the vector store which contains all the information from the game wikipedia, and provide a response.
For this you use the tool retrieve to get the relevant information.
When you receive a tool call response, use the output to format an answer to the orginal use question, provide concise, natural language responses.

- Don't repeat tool response verbatim
- Don't add supplementary information.
                                      
The query is as follows:                    
{input}

The chat history is as follows:
{chat_history}

Please provide a concise and informative response based only on the retrieved information.
If you don't know the answer, say "I don't know" (and don't provide a source).
                                      
You can use the scratchpad to store any intermediate results or notes.
The scratchpad is as follows:
{agent_scratchpad}

For every piece of information you provide, also provide the source.

Return text as follows:

<Answer to the question>
Source: source_url
""")


@tool
def retrieve(query: str): # Docstring needed for @tool, llm reads this information - see it as a part of the system prompt
    """Retrieve information from the OSRS wiki database using similarity search.
    
    Use this tool to search for information about OSRS topics including:
    - Items, weapons, armor
    - Monsters, bosses, NPCs  
    - Quests, skills, training methods
    - Locations, mechanics, strategies
    
    Args:
        query: A search term or question about OSRS content
        
    Returns:
        Relevant information from the OSRS wiki database
    """
    retrieved_docs = vector_store.similarity_search(query, k=6) # k = amount of docs retrieved, a large number can be confusing for smaller models.
    serialized = ""
    for doc in retrieved_docs:
        serialized += f"Source: {doc.metadata['source']}\nContent: {doc.page_content}\n\n"
    
    return serialized

def clean_response(response_text):
    """Remove thinking tags and clean up the response"""
    cleaned = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
    cleaned = re.sub(r'<thinking>.*?</thinking>', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'<thought>.*?</thought>', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
    
    return cleaned.strip()


tools = [retrieve]

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=20, early_stopping_method="generate")



st.set_page_config(page_title="OSRS Wiki Chatbot")
st.title("OSRS Wiki Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

user_question = st.chat_input("Ask a question")

if user_question:

    with st.chat_message("user"):
        st.markdown(user_question)
        st.session_state.messages.append(HumanMessage(user_question))

    result = agent_executor.invoke({"input": user_question, "chat_history":st.session_state.messages})

    ai_message = result["output"]
    
    cleaned_ai_message = clean_response(ai_message)

    with st.chat_message("assistant"):
        st.markdown(cleaned_ai_message)
        st.session_state.messages.append(AIMessage(cleaned_ai_message))

