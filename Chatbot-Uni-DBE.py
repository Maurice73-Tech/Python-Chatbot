#pip install -q langchain==0.0.150 pypdf pandas matplotlib tiktoken textract transformers openai faiss-cpu
#pip install langchain --upgrade

# Import libraries
import streamlit as st
from dotenv import load_dotenv
import os
import pandas as pd
import matplotlib.pyplot as plt
#from transformers import GPT2TokenizerFast
from transformers import LlamaTokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain

# Function to configure the environment
def configure():
    load_dotenv()

# Configure the environment
configure()

# Set up the OpenAI API key as an Environment Variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Load and split the PDF document into pages
loader = PyPDFLoader("DBE_FactSheet_merged.pdf")
pages = loader.load_and_split()

# Initialize chunks with the loaded pages
chunks = pages

# Library by HuggingFace for NLP to extract text from PDF file 
import textract
doc = textract.process("DBE_FactSheet_merged.pdf")

# Write the extracted text to a file
with open('DBE_FactSheet_merged.txt', 'w') as f:
    f.write(doc.decode('utf-8'))

# Read the text from the file
with open('DBE_FactSheet_merged.txt', 'r') as f:
    text = f.read()

# Load the GPT-2 tokenizer
#tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")

# Function to count the number of tokens in a text using the GPT-2 tokenizer
def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Split the text into chunks based on token count
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 512, # Maximum number of tokens per chunk
    chunk_overlap  = 24, # Number of overlapping tokens between chunks
    length_function = count_tokens, # Function to count the number of tokens in a text
)

# Create the text chunks
chunks = text_splitter.create_documents([text])

# Generate the embeddings for the chunks
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(chunks, embeddings)

# Load the QA retrieval system with specific settings
chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), db.as_retriever())

# Initialize the chat history
chat_history = []

# Streamlit favicon
st.set_page_config(page_title="DBE Information Chatbot", page_icon=":robot_face:", layout="wide")

# Streamlit Interaction
st.image("DBE-Logo.png")

# Title
st.title('Digital Business Engineering - Information Chatbot')

# User input
query = st.text_input("Please enter your question:")

# submit button
submit = st.button('Submit')

# Handle the user's question
if query:
    # End the chatbot session if user types 'exit'
    if query.lower() == 'exit':
        st.write("Thank you for using the DBE Information Chatbot")
    else:
        # Retrieve the answer for the user's question
        result = qa({"question": query, "chat_history": chat_history})
        chat_history.append((query, result['answer']))

        # Display the conversation on the web page
        st.write(f"**User:** {query}")
        st.write(f"**Chatbot:** {result['answer']}")
