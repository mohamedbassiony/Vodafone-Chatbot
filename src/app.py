"""
Vodafone Chatbot Application

This application provides a chatbot interface for querying a Vodafone database using natural language.
It uses LangChain for processing queries and Streamlit for the user interface.
The application leverages an open-source language model based on the Llama 3.1 architecture.

The chatbot can handle both text-based queries and generate visualizations based on the data.

Key Components:
- LangChain: Used for query processing and constructing the chat flow
- Streamlit: Provides the user interface for the chatbot
- Language Model: Open-source Llama 3.1 model for natural language understanding and generation
- Database Integration: Allows querying of Vodafone's database systems

Authors: Mohamed Bassiony, Mohamad Wahba, Beshoy Ashraf Samir, Mohamad Sharqawi
Date: September 24, 2024
"""

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
from pandasai.connectors import MySQLConnector
from pandasai import SmartDataframe
import streamlit as st
from PIL import Image
import psutil

# Load environment variables from a .env file
load_dotenv()

# Initialize the model
llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)
# model = Ollama(model="llama3")


def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    """
    Initialize the database connection.

    Args:
        user (str): Database username
        password (str): Database password
        host (str): Database host
        port (str): Database port
        database (str): Database name

    Returns:
        SQLDatabase: An instance of SQLDatabase for querying
    """
    db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

def check_plotting():
    """
    Create a chain to check if the user's question requires plotting.

    Returns:
        Chain: A LangChain chain for checking if plotting is needed
    """
    template = """
    You are a pandas ai expert at a company. You are interacting with a user who is asking you questions about the company's database.
    check that the user's question needs to be potting. Take the conversation history into account.
    
    Write only the Boolean Response and nothing else. Do not wrap the Boolean Response in any other text, not even backticks or space.
    
    For example:
    Question: Plot a histogram of countries showing GDP, using different colors for each bar.
    Boolean Response:True
    Question: Create a line chart of sales over time?
    Boolean Response:True
    Question: How many employees are there?
    Boolean Response:False
    
    Your turn:
    
    Question: {question}
    Boolean Response:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    return (
        prompt
        | llm
        | StrOutputParser()
    )

def get_sql_table(db):
    """
    Create a chain to extract the relevant table name from the user's question.

    Args:
        db (SQLDatabase): The database instance

    Returns:
        Chain: A LangChain chain for getting the relevant table name
    """
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, extract the table name that would answer the user's question. Take the conversation history into account.

    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the table name and nothing else. Do not wrap the table name in any other text, not even backticks or space.
    
    For example:
    Question: which 3 artists have the most tracks?
    SQL Query:Track
    Question: Name 10 artists
    SQL Query:Artist
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    def get_schema(_):
        return db.get_table_info()
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def get_sql_chain(db):
    """
    Create a chain to generate SQL queries based on the user's question.

    Args:
        db (SQLDatabase): The database instance

    Returns:
        Chain: A LangChain chain for generating SQL queries
    """
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For example:
    Question: which 3 artists have the most tracks?
    SQL Query: SELECT ArtistId, COUNT(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC LIMIT 3;
    Question: Name 10 artists
    SQL Query: SELECT Name FROM Artist LIMIT 10;
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    def get_schema(_):
        return db.get_table_info()
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    """
    Generate a response to the user's query using the database and chat history.

    Args:
        user_query (str): The user's question
        db (SQLDatabase): The database instance
        chat_history (list): The conversation history

    Returns:
        str: The generated response
    """
    sql_chain = get_sql_chain(db)
    
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    try:
        return chain.invoke({
            "question": user_query,
            "chat_history": chat_history,
        })
    except Exception as e:
        return f"Sorry, I couldn't find any data related to your question. Please try asking something else."

def table_exists(db, table_name):
    """
    Check if a table exists in the database.

    Args:
        db (SQLDatabase): The database instance
        table_name (str): The name of the table to check

    Returns:
        bool: True if the table exists, False otherwise
    """
    return table_name in db.get_table_names()

def init_MySQLConnector_pandasai(user: str, password: str, host: str, port: str, database: str, table: str) -> MySQLConnector:
    """
    Initialize the MySQL connection using MySQLConnector from pandasai.

    Args:
        user (str): Database username
        password (str): Database password
        host (str): Database host
        port (str): Database port
        database (str): Database name
        table (str): Table name

    Returns:
        MySQLConnector: An instance of MySQLConnector for pandasai
    """
    return MySQLConnector(
        config={
            "host": host,
            "port": int(port),
            "database": database,
            "username": user,
            "password": password,
            "table": table
        }
    )

def display_monitoring_metrics():
    """
    Display simple monitoring metrics in the Streamlit sidebar.
    
    This function shows CPU usage, RAM usage, and a placeholder for the number of DB queries.
    """
    st.sidebar.subheader("System Monitoring")
    
    # CPU Usage
    cpu_usage = psutil.cpu_percent(interval=1)
    st.sidebar.metric("CPU Usage", f"{cpu_usage:.2f}%")
    
    # RAM Usage
    ram = psutil.virtual_memory()
    ram_usage = ram.percent
    st.sidebar.metric("RAM Usage", f"{ram_usage:.2f}%")
    

# Initialize chat history if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello 👋! How can I assist you?"),
    ]

# Set up the Streamlit page
st.set_page_config(page_title="Vodafone Chatbot", page_icon=":speech_balloon:")
st.title("Chat with Vodafone Database")

# Sidebar for database connection settings
with st.sidebar:
    st.image("https://1000logos.net/wp-content/uploads/2017/06/Vodafone_Logo.png")
    st.subheader("Settings")
    st.write("Connect Vodafone local database and start chatting.")
    
    st.text_input("User", value="root", key="User")
    st.text_input("Password", type="password", value="root123", key="Password")
    st.text_input("Host", value="localhost", key="Host")
    st.text_input("Port", value="3306", key="Port")
    st.text_input("Database", value="Chinook", key="Database")
    
    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            db = init_database(
                st.session_state["User"],
                st.session_state["Password"],
                st.session_state["Host"],
                st.session_state["Port"],
                st.session_state["Database"]
            )
            st.session_state.db = db
            st.success("Connected to database!")
    
    # Display monitoring metrics
    display_monitoring_metrics()

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI", avatar="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTJ9uo29rKi1lXepifVFHiXtetcFLN7dyZhcQ&s"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

# Handle user input
user_query = st.chat_input("Type a message...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
        
    with st.chat_message("AI", avatar="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTJ9uo29rKi1lXepifVFHiXtetcFLN7dyZhcQ&s"):
        
        check_plotting_chain = check_plotting()
        st.session_state.boolean_plotting = check_plotting_chain.invoke({
            "question": user_query
        })

        if st.session_state.boolean_plotting == "True":
        
            sql_chain = get_sql_table(st.session_state.db)
            st.session_state.table_name = sql_chain.invoke({
                "chat_history": st.session_state.chat_history,
                "question": user_query
            })
            
            if table_exists(st.session_state.db, st.session_state.table_name):

                st.session_state.my_connector = init_MySQLConnector_pandasai(
                    st.session_state["User"],
                    st.session_state["Password"],
                    st.session_state["Host"],
                    st.session_state["Port"],
                    st.session_state["Database"],
                    st.session_state["table_name"],
                )

                # Initialize the SmartDataframe with the updated connector
                df_connector = SmartDataframe(st.session_state.my_connector, config={"llm": llm})

                response = df_connector.chat(user_query)

                image_path = r'C:\Users\moham\Desktop\Projects\Vodafone-Chatbot\src\exports\charts\temp_chart.png'
                # Open the image using PIL
                image = Image.open(image_path)

                # Display the image in Streamlit
                st.image(image)

                st.session_state.chat_history.append(AIMessage(content=response))

                with open(image_path, "rb") as file:
                    btn = st.download_button(
                        label="Download image",
                        data=file,
                        file_name=image_path,
                        mime="image/png"
                    )

            else:
                fallback_response = "I'm sorry, but I couldn't find the information you're looking for in the database."
                st.markdown(fallback_response)
                st.session_state.chat_history.append(AIMessage(content=fallback_response))

        else:
            response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
            st.markdown(response)
            st.session_state.chat_history.append(AIMessage(content=response))
