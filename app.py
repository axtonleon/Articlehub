import os
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel
import markdown
import logging

# Import for PGVector
from langchain_postgres import PGVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sqlalchemy import create_engine, text

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the API key
api_key = os.getenv("GOOGLE_API_KEY")

# Database connection details from environment variables
# IMPORTANT: You need to set these environment variables or create a .env file
# Example .env content:
# DATABASE_URL="postgresql+psycopg://user:password@host:port/database_name"
# GOOGLE_API_KEY="your_google_api_key"
CONNECTION_STRING = os.getenv("DATABASE_URL")
COLLECTION_NAME = "demo_collection" # A name for your vector collection

if not CONNECTION_STRING:
    logger.error("DATABASE_URL environment variable not set. Please set it to your PostgreSQL connection string.")
    exit(1)

#Convert CSV to a single text file
# def csv_to_txt(csv_path, txt_path):
#     df = pd.read_csv(csv_path, encoding='latin1')
#     with open(txt_path, 'w', encoding='utf-8') as f:
#         for index, row in df.iterrows():
#             # Assuming the article content is in a column named 'Content'
#             # and title in a column named 'Title'
#             f.write(f"Title: {row['Title']}\n")
#             f.write(f"Author: {row['Author']}\n")
#             f.write(f"Category: {row['Category']}\n")
#             f.write(f"Date: {row['Date']}\n")
#             f.write(f"Content: {row['Content']}\n\n")
#             print(f"Converting {csv_path} to {txt_path}")


# Paths
csv_path = "demo.csv"
txt_path = "demo.txt"


# # Convert CSV to TXT
# csv_to_txt(csv_path, txt_path)


# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

# Function to initialize and populate PGVector
def initialize_pgvector_store():
    logger.info("Initializing PGVector store...")

    # Check if the collection already exists and has data
    try:
        engine = create_engine(CONNECTION_STRING)
        with engine.connect() as connection:
            # This is a simplified check. A more robust check might query the langchain_pg_collections table
            # or count entries in the specific collection's table if you know its name.
            # For now, we'll assume if the table exists and has rows, it's populated.
            # This requires knowing the table name created by PGVector, which is usually
            # based on the collection_name.
            # Let's try to query the collection directly.
            # PGVector creates a table named 'langchain_pg_embedding' and 'langchain_pg_collection'
            # We can check if our specific collection has any entries.
            result = connection.execute(text(f"SELECT COUNT(*) FROM langchain_pg_embedding WHERE collection_id = (SELECT uuid FROM langchain_pg_collection WHERE name = '{COLLECTION_NAME}')")).scalar()
            if result and result > 0:
                logger.info(f"Collection '{COLLECTION_NAME}' already exists and is populated with {result} entries. Skipping data loading.")
                return PGVector(
                    collection_name=COLLECTION_NAME,
                    connection=CONNECTION_STRING,
                    embedding_function=embeddings
                )
    except Exception as e:
        logger.warning(f"Could not check existing PGVector collection (might not exist yet): {e}")
        # If an error occurs, it likely means the tables don't exist, so proceed to create/populate.

    logger.info("Collection not found or empty. Loading documents and populating PGVector...")

    # Load the text file and create documents
    loader = TextLoader(txt_path, encoding="utf-8")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)

    # Create and populate the PGVector store
    db = PGVector.from_documents(
        embedding=embeddings,
        documents=docs,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
    )
    logger.info("PGVector store populated successfully.")
    return db

# Initialize the PGVector store
db = initialize_pgvector_store()

# Create the FastAPI app
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Define the prompt
prompt_template = """
Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
Context:\n {context}?\n
Question: \n{input}\n
Answer:
"""
prompt = PromptTemplate.from_template(prompt_template)

# Create the QA chain
model = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.3)
retriever = db.as_retriever()
document_chain = create_stuff_documents_chain(model, prompt)
chain = create_retrieval_chain(retriever, document_chain)

class Query(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(query: Query):
    logger.info(f"Received query: {query.message}")
    try:
        response = chain.invoke({"input": query.message})
        answer = response.get("answer", "No answer found.")
        html_answer = markdown.markdown(answer)
        logger.info(f"Extracted answer: {answer}")
        return JSONResponse(content={"answer": html_answer})
    except Exception as e:
        logger.error(f"Error in /chat endpoint: {e}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)
