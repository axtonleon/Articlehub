import os
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel
import markdown

load_dotenv()

# Load the API key
api_key = os.getenv("GOOGLE_API_KEY")

# Convert CSV to a single text file
# def csv_to_txt(csv_path, txt_path):
#     df = pd.read_csv(csv_path)
#     with open(txt_path, 'w', encoding='utf-8') as f:
#         for index, row in df.iterrows():
#             # Assuming the article content is in a column named 'Content'
#             # and title in a column named 'Title'
#             f.write(f"Title: {row['Title']}\n")
#             f.write(f"Author: {row['Author']}\n")
#             f.write(f"Category: {row['Category']}\n")
#             f.write(f"Date: {row['Date']}\n")
#             f.write(f"Content: {row['Content']}\n\n")

# Paths
# csv_path = "scraped_articles_nannews.csv"
txt_path = "nannews_articles.txt"
faiss_index_path = "faiss_index"

# # Convert CSV to TXT
# csv_to_txt(csv_path, txt_path)

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

# Check if FAISS index exists, if not create it
if not os.path.exists(faiss_index_path):
    print("FAISS index not found. Creating new index...")

    # Load the text file and create FAISS index
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import TextLoader

    loader = TextLoader(txt_path, encoding="utf-8")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)

    db = FAISS.from_documents(docs, embeddings)
    db.save_local(faiss_index_path)
    print("FAISS index created and saved.")
else:
    print("FAISS index found. Loading existing index...")

# Load the FAISS index
db = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)


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

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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