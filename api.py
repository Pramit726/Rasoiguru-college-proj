from fastapi import FastAPI, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uuid
from pinecone import Pinecone
from dotenv import load_dotenv
from src.RasoiGuru.components.check_index import IndexManager
from src.RasoiGuru.components.data_ingestion import DataIngestor
from src.RasoiGuru.pipeline.pipeline import create_pipeline  
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from src.utils import vector_exist, extract_answer, get_paths
import yaml

# Load environment variables from a .env file
load_dotenv()

# Load parameters from params.yml
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Access Pinecone parameters from the YAML file
pinecone_params = params.get("pinecone", {})
index_name = pinecone_params.get("index_name", "default_index")
cloud = pinecone_params.get("cloud", "aws")
region = pinecone_params.get("region", "us-east-1")

# Initialize FastAPI app
app = FastAPI(
    title="RasoiGuru",
    description="RasoiGuru is your ultimate cooking assistant chatbot, offering detailed cooking instructions, ingredient substitutions, and personalized culinary tips to elevate your kitchen skills."
)

# Define input model
class Input(BaseModel):
    query: str

# Initialize memory store for session management
memory_store = {}


# Route to welcome page
@app.get("/", summary="Welcome", tags=["Welcome"])
def welcome():
    return "Welcome to RasoiGuru"


# Function to get memory for session management
def get_memory(session_id: str):
    if session_id not in memory_store:
        memory_store[session_id] = ConversationBufferWindowMemory(k=3, return_messages=True, memory_key="chat_history")
    return memory_store[session_id]


# Route for chat functionality
@app.post("/chat", summary="Chat with RasoiGuru", tags=["Chat"], response_model=Input)
async def chat(input: Input, request: Request):
    # Generate a unique session ID for each user session
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())

    memory = get_memory(session_id)

    # Check if vector index exists, if not, create index and insert documents
    index_manager = IndexManager(index_name=index_name, cloud=cloud, region=region)
    pc = Pinecone()
    if not vector_exist(index_manager.index_name, pc):
        data_ingestor = DataIngestor()
        pdf_files = get_paths()
        docs = data_ingestor.load_documents(pdf_files)
        chunks = data_ingestor.make_chunks(docs)
    else:
        pdf_files = get_paths()
        chunks = []

    # Create or retrieve the Pinecone index
    index = index_manager.create_index()

    # Insert documents if index is newly created
    if chunks:
        vectorstores = index_manager.insert_documents(pdf_files, chunks)
    else:
        vectorstores = None  # No new documents to insert

    # Create pipeline (tools and executor)
    executor = create_pipeline(vectorstores, memory)

    # Get response
    response = executor.invoke({"input": input.query})
    response = extract_answer(result=response)
    response_data = {'input': input.query, 'output': response}

    # Set session ID cookie and return response
    response = JSONResponse(content=response_data)
    response.set_cookie(key="session_id", value=session_id)
    return response